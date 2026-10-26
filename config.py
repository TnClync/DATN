from pathlib import Path

# ============================================================
# PROJECT
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
APP_NAME = "DATN-Hybrid-Recommender"

# ============================================================
# SPARK EXECUTION
# ============================================================

MASTER = "local[*]"
SHUFFLE_PARTITIONS = 16

# ============================================================
# RAM CONFIGURATION
# ============================================================

# Dung lượng heap tối đa cho Spark Driver
# Đơn vị: GB
SPARK_DRIVER_MEMORY = "3g"

# Local mode: executor chạy trong cùng JVM với driver.
# Cấu hình này chủ yếu có ý nghĩa khi chạy cluster mode.
SPARK_EXECUTOR_MEMORY = "2g"

# Bộ nhớ dành cho Spark SQL execution và storage.
# Spark mặc định dùng khoảng 60% heap cho unified memory.
SPARK_MEMORY_FRACTION = 0.6

# Phần unified memory dành cho storage trước khi
# execution có thể mượn thêm.
SPARK_MEMORY_STORAGE_FRACTION = 0.5

# Số luồng CPU Spark được dùng.
# Đổi thành "4" nếu muốn giới hạn số luồng.
SPARK_CPU_CORES = "*"

# ============================================================
# SPARK SQL
# ============================================================

AQE_ENABLED = True
AQE_COALESCE_PARTITIONS_ENABLED = True

# ============================================================
# MODEL PATHS
# ============================================================

MODEL_DIR = PROJECT_DIR / "models" / "als"
CONTENT_BASED_DIR = PROJECT_DIR / "models" / "content_based"
BPR_MF_DIR = PROJECT_DIR / "models" / "bpr_mf"

USER_FACTORS_PATH = MODEL_DIR / "userFactors"
ITEM_FACTORS_PATH = MODEL_DIR / "itemFactors"
USER_MAPPING_PATH = MODEL_DIR / "user_mapping"
ITEM_MAPPING_PATH = MODEL_DIR / "item_mapping"
TRAIN_PATH = MODEL_DIR / "train"

# ============================================================
# HYBRID MODEL PATHS
# ============================================================

HYBRID_MODEL_DIR = PROJECT_DIR / "models" / "hybrid"
HYBRID_CONFIG_PATH = HYBRID_MODEL_DIR / "hybrid_config.json"
HYBRID_WEIGHTS_PATH = HYBRID_MODEL_DIR / "hybrid_model.pt"

# Serving cache: build một lần bằng scripts/build_serving_cache.py
SERVING_DIR = PROJECT_DIR / "models" / "serving"
SERVING_META_PATH = SERVING_DIR / "catalog_meta.sqlite"
SERVING_INDEX_PATH = SERVING_DIR / "catalog_index.npz"
SERVING_INFO_PATH = SERVING_DIR / "build_info.json"

# ============================================================
# RECOMMENDATION
# ============================================================

DEFAULT_TOP_N = 50
DEFAULT_DOMAIN = "Books"
EXCLUDE_SEEN_ITEMS = True

# Số ứng viên được chấm Content-Based. Càng lớn càng chính xác nhưng chậm hơn.
# 500 là điểm cân bằng tốt trên laptop 8GB RAM.
CANDIDATE_POOL_SIZE = 500
MAX_SEEN_FOR_PROFILE = 50

# ============================================================
# DATA SCHEMA
# ============================================================

RAW_USER_ID_COL = "reviewerID"
USER_INDEX_COL = "user_idx"
ITEM_KEY_COL = "item_key"
ITEM_INDEX_COL = "item_idx"
RATING_COL = "rating"
TIMESTAMP_COL = "unixReviewTime"
DOMAIN_COL = "domain"
ITEM_KEY_SEPARATOR = "::"
