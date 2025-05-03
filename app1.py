import os
import string
import unicodedata
from datetime import datetime
from pprint import pprint
import uuid
import torch
import torchaudio
import time
import numpy as np
from flask import Flask, request, render_template, send_file, jsonify
from werkzeug.utils import secure_filename
from tqdm import tqdm
from underthesea import sent_tokenize
from unidecode import unidecode
import concurrent.futures
import threading

try:
    from vinorm import TTSnorm
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
except Exception as e:
    print(f"Lỗi khi import thư viện: {e}")

app = Flask(__name__)

# Cấu hình thư mục lưu trữ
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
ALLOWED_EXTENSIONS = {'mp3', 'wav'}

# Đảm bảo thư mục tồn tại
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Giới hạn 16MB cho tệp tải lên

# Tối ưu hóa CUDA nếu khả dụng
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# Biến global để lưu trữ model đã tải
global_model = None
model_lock = threading.Lock()  # Lock để đảm bảo thread-safety khi truy cập model

def allowed_file(filename):
    """Kiểm tra xem tệp có đuôi hợp lệ hay không"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load model
def clear_gpu_cache():
    """Giải phóng bộ nhớ GPU nếu có"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_model(xtts_checkpoint, xtts_config, xtts_vocab):
    """Tải mô hình XTTS từ các file checkpoint, config và vocab"""
    global global_model
    
    # Nếu model đã được tải, không tải lại
    if global_model is not None:
        return global_model
        
    with model_lock:
        # Kiểm tra lại trong lock để tránh race condition
        if global_model is not None:
            return global_model
            
        clear_gpu_cache()
        if not xtts_checkpoint or not xtts_config or not xtts_vocab:
            return "You need to run the previous steps or manually set the model paths!"
            
        print("Loading XTTS model...")
        start_load_time = time.time()
        
        config = XttsConfig()
        config.load_json(xtts_config)
        XTTS_MODEL = Xtts.init_from_config(config)
        XTTS_MODEL.load_checkpoint(config,
                                   checkpoint_path=xtts_checkpoint,
                                   vocab_path=xtts_vocab,
                                   use_deepspeed=False)
        if torch.cuda.is_available():
            XTTS_MODEL.cuda()
        
        end_load_time = time.time()
        print(f"Model Loaded in {end_load_time - start_load_time:.2f} seconds!")
        global_model = XTTS_MODEL
        return global_model

def get_file_name(text, max_char=50):
    """Tạo tên file từ văn bản đầu vào"""
    filename = text[:max_char]
    filename = filename.lower()
    filename = filename.replace(" ", "_")
    filename = filename.translate(str.maketrans("", "", string.punctuation.replace("_", "")))
    filename = unidecode(filename)
    current_datetime = datetime.now().strftime("%m%d%H%M%S")
    filename = f"{current_datetime}_{filename}"
    return filename

def calculate_keep_len(text, lang):
    """Tính toán độ dài giữ lại cho âm thanh"""
    if lang in ["ja", "zh-cn"]:
        return -1

    word_count = len(text.split())
    num_punct = (
        text.count(".")
        + text.count("!")
        + text.count("?")
        + text.count(",")
    )

    if word_count < 5:
        return 15000 * word_count + 2000 * num_punct
    elif word_count < 10:
        return 13000 * word_count + 2000 * num_punct
    return -1

def normalize_vietnamese_text(text):
    """Chuẩn hóa văn bản tiếng Việt"""
    try:
        text = (
            TTSnorm(text, unknown=False, lower=False, rule=True)
            .replace("..", ".")
            .replace("!.", "!")
            .replace("?.", "?")
            .replace(" .", ".")
            .replace(" ,", ",")
            .replace('"', "")
            .replace("'", "")
            .replace("AI", "Ây Ai")
            .replace("A.I", "Ây Ai")
            .replace("+", "cộng")
            .replace("-", "trừ")
            .replace("*", "nhân")
            .replace("/", "chia")
            .replace("=", "bằng")
        )
        return text
    except Exception as e:
        print(f"Lỗi khi chuẩn hóa văn bản: {e}")
        return text

def process_text_chunk(args):
    """Hàm xử lý một đoạn văn bản để chạy song song"""
    model, text, lang, gpt_cond_latent, speaker_embedding = args
    
    if text.strip() == "":
        return None
    
    with torch.no_grad():  # Tiết kiệm bộ nhớ
        wav_chunk = model.inference(
            text=text,
            language=lang,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=0.3,
            length_penalty=1.0,
            repetition_penalty=10.0,
            top_k=30,
            top_p=0.85,
        )
    
    # Quick hack for short sentences
    keep_len = calculate_keep_len(text, lang)
    if keep_len > 0:
        wav_chunk["wav"] = wav_chunk["wav"][:keep_len]
    
    return wav_chunk["wav"]

def run_tts_parallel(model, lang, tts_text, speaker_audio_file, 
                    normalize_text=True, verbose=False, output_chunks=False,
                    num_workers=4):
    """Phiên bản song song của hàm run_tts sử dụng đa luồng"""
    start_time = time.time()
    
    if model is None or not speaker_audio_file:
        return "You need to run the previous step to load the model!", None, None, None

    output_dir = OUTPUT_FOLDER
    os.makedirs(output_dir, exist_ok=True)

    # Lấy latent embedding từ audio tham chiếu
    start_cond_time = time.time()
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=speaker_audio_file,
        gpt_cond_len=model.config.gpt_cond_len,
        max_ref_length=model.config.max_ref_len,
        sound_norm_refs=model.config.sound_norm_refs,
    )
    end_cond_time = time.time()
    cond_time = end_cond_time - start_cond_time
    
    # Chuẩn hóa văn bản tiếng Việt nếu cần
    if normalize_text and lang == "vi":
        tts_text = normalize_vietnamese_text(tts_text)

    # Phân đoạn văn bản thành các câu
    if lang in ["ja", "zh-cn"]:
        tts_texts = tts_text.split("。")
    else:
        tts_texts = sent_tokenize(tts_text)

    if verbose:
        print(f"Tổng số câu cần xử lý: {len(tts_texts)}")
        print("Text for TTS:")
        pprint(tts_texts)

    # Bắt đầu đo thời gian cho việc inference
    start_inference_time = time.time()
    
    # Chuẩn bị tham số cho các tác vụ song song
    valid_texts = [text for text in tts_texts if text.strip() != ""]
    tasks = [(model, text, lang, gpt_cond_latent, speaker_embedding) for text in valid_texts]
    
    # Xử lý song song
    wav_chunks = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_text_chunk, task): i for i, task in enumerate(tasks)}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc="Processing"):
            idx = futures[future]
            try:
                result = future.result()
                if result is not None:
                    wav_chunks.append(result)
                    
                    if output_chunks and verbose:
                        out_path = os.path.join(output_dir, f"{get_file_name(valid_texts[idx])}.wav")
                        torchaudio.save(out_path, torch.tensor(result).unsqueeze(0), 24000)
                        print(f"Saved chunk to {out_path}")
            except Exception as e:
                print(f"Error processing chunk {idx}: {e}")
    
    end_inference_time = time.time()
    inference_time = end_inference_time - start_inference_time
    
    # Bắt đầu đo thời gian cho việc lưu file
    start_save_time = time.time()
    
    # Kết hợp tất cả các đoạn âm thanh
    if not wav_chunks:
        return None, cond_time, inference_time, 0
    
    out_wav = np.concatenate(wav_chunks)
    out_wav_tensor = torch.tensor(out_wav).unsqueeze(0)
    out_filename = f"{get_file_name(tts_text)}.wav"
    out_path = os.path.join(output_dir, out_filename)
    torchaudio.save(out_path, out_wav_tensor, 24000)

    end_save_time = time.time()
    save_time = end_save_time - start_save_time
    
    total_time = end_save_time - start_time
    if verbose:
        print(f"Saved final file to {out_path}")
        print(f"Thống kê thời gian:")
        print(f"  Conditioning: {cond_time:.2f} giây")
        print(f"  Inference: {inference_time:.2f} giây")
        print(f"  Lưu file: {save_time:.2f} giây")
        print(f"  Tổng cộng: {total_time:.2f} giây")
    
    return out_path, cond_time, inference_time, save_time

@app.route('/')
def index():
    """Hiển thị trang chủ với form nhập liệu"""
    return render_template('index1.html')

@app.route('/synthesize', methods=['POST'])
def synthesize():
    """API để chuyển đổi văn bản thành giọng nói"""
    global global_model
    
    if 'voice_file' not in request.files:
        return jsonify({'error': 'Không tìm thấy tệp giọng nói'}), 400
    
    file = request.files['voice_file']
    if file.filename == '':
        return jsonify({'error': 'Không có tệp nào được chọn'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Định dạng tệp không được hỗ trợ. Chỉ chấp nhận .mp3 và .wav'}), 400
    
    text = request.form.get('text', '')
    if not text:
        return jsonify({'error': 'Vui lòng nhập văn bản để chuyển đổi'}), 400
    
    language = request.form.get('language', 'vi')
    num_workers = int(request.form.get('num_workers', '4'))
    normalize = request.form.get('normalize', 'true').lower() == 'true'
    
    # Lưu tệp giọng nói tải lên
    filename = secure_filename(file.filename)
    voice_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{str(uuid.uuid4())}_{filename}")
    file.save(voice_file_path)
    
    try:
        # Đảm bảo model đã được tải
        if global_model is None:
            global_model = load_model(
                xtts_checkpoint="model/model.pth",
                xtts_config="model/config.json",
                xtts_vocab="model/vocab.json"
            )
        
        # Chạy TTS song song
        output_file, cond_time, inference_time, save_time = run_tts_parallel(
            model=global_model,
            lang=language,
            tts_text=text,
            speaker_audio_file=voice_file_path,
            normalize_text=normalize,
            verbose=True,
            num_workers=num_workers
        )
        
        if output_file is None:
            return jsonify({'error': 'Không thể tổng hợp giọng nói từ văn bản đã cho'}), 500
        
        # Chuyển sang MP3 nếu yêu cầu
        output_mp3 = None
        if request.form.get('output_format', 'wav') == 'mp3':
            output_mp3 = output_file.replace('.wav', '.mp3')
            try:
                os.system(f"ffmpeg -i {output_file} -vn -ar 44100 -ac 2 -b:a 192k {output_mp3} -y -loglevel quiet")
                if os.path.exists(output_mp3):
                    os.remove(output_file)  # Xóa file wav tạm nếu chuyển đổi thành công
                    output_file = output_mp3
            except Exception as e:
                print(f"Lỗi khi chuyển sang MP3: {e}")
        
        # Trả về thông tin về tệp âm thanh đã tạo
        return jsonify({
            'success': True,
            'message': 'Chuyển đổi thành công',
            'audio_file': os.path.basename(output_file),
            'stats': {
                'conditioning_time': f"{cond_time:.2f} giây",
                'inference_time': f"{inference_time:.2f} giây",
                'save_time': f"{save_time:.2f} giây",
                'total_time': f"{cond_time + inference_time + save_time:.2f} giây"
            }
        })
    except Exception as e:
        return jsonify({'error': f'Lỗi khi chuyển đổi: {str(e)}'}), 500
    finally:
        # Xóa tệp giọng nói tải lên sau khi xử lý xong
        if os.path.exists(voice_file_path):
            os.remove(voice_file_path)

@app.route('/download/<filename>')
def download_file(filename):
    """API để tải xuống tệp âm thanh đã tạo"""
    return send_file(os.path.join(app.config['OUTPUT_FOLDER'], filename), as_attachment=True)

@app.route('/api/tts', methods=['POST'])
def api_tts():
    """API để chuyển đổi văn bản thành giọng nói qua JSON"""
    global global_model
    
    # Lấy dữ liệu từ request JSON
    data = request.json
    if not data:
        return jsonify({'error': 'Không có dữ liệu được gửi'}), 400
    
    text = data.get('text')
    speaker_file = data.get('speaker_file', 'Seren2.wav')  # File tham chiếu mặc định
    language = data.get('language', 'vi')
    normalize = data.get('normalize', True)
    num_workers = data.get('num_workers', 4)  # Số luồng xử lý song song
    output_format = data.get('output_format', 'wav')
    
    if not text:
        return jsonify({'error': 'Thiếu trường "text"'}), 400
    
    # Kiểm tra file âm thanh tham chiếu
    if not os.path.exists(speaker_file):
        return jsonify({'error': f'File âm thanh tham chiếu "{speaker_file}" không tồn tại'}), 400
    
    try:
        # Đảm bảo model đã được tải
        if global_model is None:
            global_model = load_model(
                xtts_checkpoint="model/model.pth",
                xtts_config="model/config.json",
                xtts_vocab="model/vocab.json"
            )
        
        # Chạy TTS song song
        output_file, cond_time, inference_time, save_time = run_tts_parallel(
            model=global_model,
            lang=language,
            tts_text=text,
            speaker_audio_file=speaker_file,
            normalize_text=normalize,
            verbose=True,
            num_workers=num_workers
        )
        
        if output_file is None:
            return jsonify({'error': 'Không thể tổng hợp giọng nói từ văn bản đã cho'}), 500
        
        # Chuyển sang MP3 nếu yêu cầu
        if output_format == 'mp3':
            output_mp3 = output_file.replace('.wav', '.mp3')
            try:
                os.system(f"ffmpeg -i {output_file} -vn -ar 44100 -ac 2 -b:a 192k {output_mp3} -y -loglevel quiet")
                if os.path.exists(output_mp3):
                    os.remove(output_file)  # Xóa file wav tạm nếu chuyển đổi thành công
                    output_file = output_mp3
            except Exception as e:
                print(f"Lỗi khi chuyển sang MP3: {e}")
        
        # Tạo URL để tải xuống tệp
        download_url = f"/download/{os.path.basename(output_file)}"
        
        return jsonify({
            'success': True,
            'message': 'Chuyển đổi thành công',
            'download_url': download_url,
            'stats': {
                'conditioning_time': f"{cond_time:.2f} giây",
                'inference_time': f"{inference_time:.2f} giây",
                'save_time': f"{save_time:.2f} giây",
                'total_time': f"{cond_time + inference_time + save_time:.2f} giây"
            }
        })
    except Exception as e:
        return jsonify({'error': f'Lỗi khi chuyển đổi: {str(e)}'}), 500

@app.route('/status', methods=['GET'])
def status():
    """Kiểm tra trạng thái của server và GPU"""
    global global_model
    
    # Kiểm tra mô hình đã được tải chưa
    model_loaded = global_model is not None
    
    # Kiểm tra GPU available hay không
    gpu_available = torch.cuda.is_available()
    gpu_info = {}
    
    if gpu_available:
        gpu_info = {
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(0),
            "memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**2:.2f} MB",
            "memory_reserved": f"{torch.cuda.memory_reserved(0) / 1024**2:.2f} MB"
        }
    
    return jsonify({
        "status": "running",
        "model_loaded": model_loaded,
        "gpu_available": gpu_available,
        "gpu_info": gpu_info
    })

if __name__ == '__main__':
    # Đảm bảo thư mục output tồn tại
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Tải mô hình trước khi khởi động server
    print("> Đang nạp mô hình...")
    try:
        global_model = load_model(
            xtts_checkpoint="model/model.pth",
            xtts_config="model/config.json",
            xtts_vocab="model/vocab.json"
        )
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        print("Server sẽ tải mô hình khi nhận yêu cầu đầu tiên.")
    
    # Khởi động Flask server
    app.run(host='0.0.0.0', port=9321, debug=False, threaded=True)