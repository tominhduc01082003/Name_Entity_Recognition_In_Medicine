import os
import urllib.request

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
VNCORE_DIR = os.path.join(PROJECT_ROOT, "vncorenlp")
MODEL_DIR = os.path.join(VNCORE_DIR, "models", "wordsegmenter")

os.makedirs(MODEL_DIR, exist_ok=True)

files_to_download = {
    "VnCoreNLP-1.2.jar.1": "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.2.jar.1",
    "vi-vocab": "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab",
    "wordsegmenter.rdr": "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr"
}

for filename, url in files_to_download.items():
    if filename == "VnCoreNLP-1.2.jar.1":
        dest = os.path.join(VNCORE_DIR, filename)
    else:
        dest = os.path.join(MODEL_DIR, filename)

    if not os.path.exists(dest):
        print(f"Đang tải {filename} từ {url}...")
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"Đã lưu: {dest}")
        except Exception as e:
            print(f"Lỗi khi tải {filename}: {e}")
    else:
        print(f"{filename} đã tồn tại, bỏ qua.")

print("\nHoàn tất tải và sắp xếp file VnCoreNLP.")
print(f"Cấu trúc thư mục:\n{VNCORE_DIR}")
