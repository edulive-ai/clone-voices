# Speech2Text 

## 🚀 Mục tiêu

Chuyển đổi audio sang text, clone giọng nói bất kì

---

## 📦 1. Tạo môi trường ảo

### ✅ Option 1: Dùng Conda

```bash
conda create -n stt python=3.10 -y
conda activate stt
```

### ✅ Option 2: Dùng venv(Python chuẩn)

```bash
python3 -m venv stt
source stt/bin/activate  # Trên macOS/Linux
stt\Scripts\activate     # Trên Windows
```
## 📦 2. Cài đặt thư viện

```bash
sudo apt install fmpeg ## cai dat fmpeg
pip install -r requirements.txt

```
## 📦 3. setup
```bash
python3 setup_vixtts.py # ubuntu/linux
python setup_vixtts.py # windows 
```
## 4. Run
```bash
python3 app.py # ubuntu/linux
python app.py # windows 
```




