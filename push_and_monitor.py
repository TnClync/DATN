import json
import requests
from pathlib import Path

TOKEN = "KGAT_089002d2153e86bd4c67f426b97866c3"
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

BASE_DIR = Path("/home/clyde/DATN/kaggle_train")
MODEL_NAME = "multiaffinity-hybrid-2gpu"
MODEL_DIR = BASE_DIR / MODEL_NAME
meta_file = MODEL_DIR / "kernel-metadata.json"

print("=== BẮT ĐẦU CHẨN ĐOÁN VÀ ĐẨY CODE TỰ ĐỘNG ===")

# 1. Kiểm tra thông tin tài khoản qua API Kaggle
print("🔍 Đang kiểm tra quyền hạn và username từ Token...")
test_res = requests.get("https://www.kaggle.com/api/v1/kernels/list?page=1&pageSize=1", headers=HEADERS)
print(f"Mã phản hồi xác thực Token: {test_res.status_code}")

if test_res.status_code != 200:
    print("❌ Token không hợp lệ hoặc lỗi xác thực:", test_res.text)
    exit(1)

# 2. Đọc metadata
with open(meta_file, "r", encoding="utf-8") as f:
    meta = json.load(f)

current_id = meta.get("id", "")
print(f"📌 ID hiện tại trong metadata: {current_id}")

# Tự động thử sửa username nếu có sự nhầm lẫn giữa minhteonguyn và nguynteo
possible_ids = [current_id]
if "minhteonguyn/" in current_id:
    possible_ids.append(current_id.replace("minhteonguyn/", "nguynteo/"))
elif "nguynteo/" in current_id:
    possible_ids.append(current_id.replace("nguynteo/", "minhteonguyn/"))

notebook_path = MODEL_DIR / meta.get("code_file", f"{MODEL_NAME}.ipynb")
if not notebook_path.exists():
    raise FileNotFoundError(f"Không tìm thấy file notebook tại: {notebook_path}")

notebook_content = notebook_path.read_text(encoding="utf-8")

success = False
for test_id in possible_ids:
    print(f"\n🚀 Đang thử push với slug: {test_id} ...")
    payload = {
        "slug": test_id,
        "newTitle": meta["title"],
        "text": notebook_content,
        "language": meta.get("language", "python"),
        "kernelType": meta.get("kernel_type", "notebook"),
        "isPrivate": str(meta.get("is_private", "true")).lower() == "true",
        "enableGpu": str(meta.get("enable_gpu", "true")).lower() == "true",
        "enableTpu": str(meta.get("enable_tpu", "false")).lower() == "true",
        "enableInternet": str(meta.get("enable_internet", "true")).lower() == "true",
        "datasetDataSources": meta.get("dataset_sources", []),
        "kernelDataSources": meta.get("kernel_sources", [])
    }

    response = requests.post(
        "https://www.kaggle.com/api/v1/kernels/push",
        headers=HEADERS,
        json=payload
    )

    print(f"Mã phản hồi HTTP: {response.status_code}")
    if response.status_code == 200:
        res_data = response.json()
        print("🎉 Đẩy code thành công rực rỡ!")
        print(f"📦 Version số: {res_data.get('versionNumber', 'N/A')}")
        print(f"🔗 Link theo dõi: https://www.kaggle.com/code/{test_id}")
        
        # Cập nhật lại file metadata cho đúng với ID đã push thành công
        meta["id"] = test_id
        meta_file.write_text(json.dumps(meta, indent=2))
        print( "đã được cập nhật tự động trong file metadata!")
        success = True
        break
    else:
        print(f"❌ Thất bại với {test_id}: {response.text}")

if not success:
    print("\n❌ Tất cả các thử nghiệm đều thất bại. Vui lòng kiểm tra lại slug trên trang web Kaggle.")