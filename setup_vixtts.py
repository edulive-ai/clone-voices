import os
import subprocess

def run_command(command):
    try:
        print(f"Đang chạy: {command}")
        subprocess.run(command, shell=True, check=True)
        print(f"✅ Đã hoàn thành: {command}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Lỗi khi chạy: {command}")
        print(f"Chi tiết lỗi: {e}")
        return False
    return True

def install_libraries():
    print("🚀 Bắt đầu quá trình cài đặt thư viện...")
    
    # Xóa thư mục TTS nếu đã tồn tại
    if os.path.exists("TTS"):
        run_command("rm -rf TTS/")
    
    # Clone repo TTS với nhánh hỗ trợ tiếng Việt
    if not run_command("git clone --branch add-vietnamese-xtts -q https://github.com/thinhlpg/TTS.git"):
        return False
    
    # Cài đặt thư viện TTS từ repo
    if not run_command("pip install --use-deprecated=legacy-resolver -q -e TTS"):
        return False
    if not run_command("pip install -q vinorm==2.0.7"):
        return False
    if not run_command("pip install -q cutlet"):
        return False
    if not run_command("pip install -q unidic==1.1.0"):
        return False
    if not run_command("pip install -q underthesea"):
        return False
    if not run_command("pip install -q gradio==4.35"):
        return False
    if not run_command("pip install deepfilternet==0.5.6 -q"):
        return False
    if not run_command("pip install numpy==1.26.4"):
        return False
    
    
    print("\n✅ Cài đặt hoàn tất!")
    print("Bạn có thể sử dụng vixTTS để tạo giọng nói tiếng Việt.")
    return True


if __name__ == "__main__":
    install_libraries()
    from huggingface_hub import snapshot_download
    os.system("python -m unidic download")
    print(" > Tải mô hình...")
    snapshot_download(repo_id="thinhlpg/viXTTS",
                    repo_type="model",
                    local_dir="model")
