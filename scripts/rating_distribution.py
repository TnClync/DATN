from pyspark.sql import SparkSession, functions as F
from pathlib import Path
import json
import tempfile

# =========================
# CONFIG
# =========================
PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = str(PROJECT_DIR / "data" / "Interactions_indexed")
OUTPUT_PATH = str(PROJECT_DIR / "outputs" / "rating_distribution.json")
SPARK_TEMP = str(Path(tempfile.gettempdir()) / "datn-spark")

# =========================
# SPARK SESSION
# =========================
Path(SPARK_TEMP).mkdir(parents=True, exist_ok=True)

spark = (
    SparkSession.builder
    .appName("CountDistinctRatings")
    .master("local[*]")
    .config("spark.local.dir", SPARK_TEMP)
    .config("spark.sql.shuffle.partitions", "16")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


def main():
    if not Path(INPUT_PATH).exists():
        raise FileNotFoundError(
            f"Không tìm thấy dữ liệu: {INPUT_PATH}"
        )

    interactions = spark.read.parquet(INPUT_PATH)

    if "rating" not in interactions.columns:
        raise ValueError("Dataset không có cột rating.")

    # Bỏ interaction không có rating
    valid_ratings = interactions.filter(
        F.col("rating").isNotNull()
    )

    total = valid_ratings.count()

    # Đếm số lượng và tỷ lệ của từng rating
    rating_counts = (
        valid_ratings
        .groupBy("rating")
        .agg(F.count("*").alias("count"))
        .withColumn(
            "percentage",
            F.round(
                F.col("count") * 100.0 / F.lit(total),
                2
            )
        )
        .orderBy("rating")
    )

    # Collect chỉ bảng tổng hợp rating nhỏ,
    # không collect toàn bộ interactions.
    rows = rating_counts.collect()

    result = {
        "total_interactions_with_rating": total,
        "distinct_rating_count": len(rows),
        "rating_distribution": [
            {
                "rating": row["rating"],
                "count": row["count"],
                "percentage": row["percentage"]
            }
            for row in rows
        ]
    }

    # Ghi thành một file JSON duy nhất
    output_file = Path(OUTPUT_PATH)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\n===== RATING DISTRIBUTION =====")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nSaved JSON to: {OUTPUT_PATH}")


if __name__ == "__main__":
    try:
        main()
    finally:
        spark.stop()
