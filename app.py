import os
import torch
import torchaudio
import numpy as np
import time
import uuid
from flask import Flask, request, render_template, send_file, jsonify
from werkzeug.utils import secure_filename
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

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

# Đường dẫn đến thư mục chứa model
MODEL_PATH = "model"

# Khởi tạo model
print("Đang tải mô hình...")
start_load_time = time.time()
config = XttsConfig()
config.load_json(os.path.join(MODEL_PATH, "config.json"))
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=MODEL_PATH)
end_load_time = time.time()
time_load_model = end_load_time - start_load_time
print(f"Đã tải mô hình thành công trong {end_load_time - start_load_time:.2f} giây")

# Dùng GPU nếu có
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Đang sử dụng thiết bị: {device}")
model.to(device)

def allowed_file(filename):
    """Kiểm tra xem tệp có đuôi hợp lệ hay không"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def text_to_speech(text, speaker_wav, output_path, language="vi"):
    """Chuyển đổi văn bản thành giọng nói, sử dụng mẫu giọng từ file audio"""
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    # Bắt đầu đo thời gian cho việc tạo conditioning latents
    start_cond_time = time.time()
    # Tạo conditioning latents từ file giọng mẫu
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=speaker_wav,
        gpt_cond_len=model.config.gpt_cond_len,
        max_ref_length=model.config.max_ref_len,
        sound_norm_refs=model.config.sound_norm_refs,
    )
    end_cond_time = time.time()
    cond_time = end_cond_time - start_cond_time
    print(f"Thời gian tạo conditioning latents: {cond_time:.2f} giây")
    
    # Bắt đầu đo thời gian cho việc inference
    start_inference_time = time.time()
    # Tạo wav từ văn bản
    outputs = model.inference(
        text=text,
        language=language,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=0.3,
        length_penalty=1.0,
        repetition_penalty=10.0,
        top_k=30,
        top_p=0.85,
    )
    end_inference_time = time.time()
    inference_time = end_inference_time - start_inference_time
    print(f"Thời gian inference: {inference_time:.2f} giây")
    
    # Bắt đầu đo thời gian cho việc lưu file
    start_save_time = time.time()
    # Lưu file WAV
    wav_path = output_path
    if output_path.endswith('.mp3'):
        wav_path = output_path.replace('.mp3', '.wav')
    
    # Chuyển numpy array sang tensor torch
    wav_tensor = torch.tensor(outputs["wav"]).unsqueeze(0)
    torchaudio.save(wav_path, wav_tensor, 24000)
    
    # Chuyển sang MP3 nếu cần
    if output_path.endswith('.mp3'):
        try:
            os.system(f"ffmpeg -i {wav_path} -vn -ar 44100 -ac 2 -b:a 192k {output_path} -y -loglevel quiet")
            if os.path.exists(output_path):
                os.remove(wav_path)  # Xóa file wav tạm nếu chuyển đổi thành công
                end_save_time = time.time()
                save_time = end_save_time - start_save_time
                print(f"Thời gian lưu file: {save_time:.2f} giây")
                return output_path, cond_time, inference_time, save_time
        except Exception as e:
            end_save_time = time.time()
            save_time = end_save_time - start_save_time
            print(f"Lỗi khi chuyển sang MP3: {e}")
            print(f"Thời gian lưu file: {save_time:.2f} giây")
            return wav_path, cond_time, inference_time, save_time
    
    end_save_time = time.time()
    save_time = end_save_time - start_save_time
    print(f"Thời gian lưu file: {save_time:.2f} giây")
    return wav_path, cond_time, inference_time, save_time

@app.route('/')
def index():
    """Hiển thị trang chủ với form nhập liệu"""
    return render_template('index.html')

@app.route('/synthesize', methods=['POST'])
def synthesize():
    """API để chuyển đổi văn bản thành giọng nói"""
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
    
    # Lưu tệp giọng nói tải lên
    filename = secure_filename(file.filename)
    voice_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{str(uuid.uuid4())}_{filename}")
    file.save(voice_file_path)
    
    # Tạo tên tệp đầu ra duy nhất
    output_filename = f"output_{str(uuid.uuid4())}.mp3"
    output_file_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    
    try:
        # Chuyển đổi văn bản thành giọng nói
        result_file, cond_time, inference_time, save_time = text_to_speech(
            text=text,
            speaker_wav=voice_file_path,
            output_path=output_file_path,
            language=language
        )
        
        # Trả về thông tin về tệp âm thanh đã tạo
        return jsonify({
            'success': True,
            'message': 'Chuyển đổi thành công',
            'audio_file': output_filename,
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
    """API cho phép sử dụng qua các ứng dụng khác"""
    # Kiểm tra dữ liệu đầu vào
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
    
    # Lưu tệp giọng nói tải lên
    filename = secure_filename(file.filename)
    voice_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{str(uuid.uuid4())}_{filename}")
    file.save(voice_file_path)
    
    # Tạo tên tệp đầu ra duy nhất
    output_filename = f"output_{str(uuid.uuid4())}.mp3"
    output_file_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    
    try:
        # Chuyển đổi văn bản thành giọng nói
        result_file, cond_time, inference_time, save_time = text_to_speech(
            text=text,
            speaker_wav=voice_file_path,
            output_path=output_file_path,
            language=language
        )
        
        # Tạo URL để tải xuống tệp
        download_url = f"/download/{os.path.basename(result_file)}"
        
        return jsonify({
            'success': True,
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
    finally:
        # Xóa tệp giọng nói tải lên sau khi xử lý xong
        if os.path.exists(voice_file_path):
            os.remove(voice_file_path)

if __name__ == '__main__':
    # Chạy ứng dụng Flask trên cổng 9322
    app.run(host='0.0.0.0', port=9321, debug=True)
