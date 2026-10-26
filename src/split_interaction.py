
from pathlib import Path
import shutil
import sys
import tempfile

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = Path(
    PROJECT_DIR / "data" / "interactions.parquet"
)

OUTPUT_DIR = Path(
    PROJECT_DIR / "data" / "splits"
)

SPARK_TMP = str(Path(tempfile.gettempdir()) / "datn-spark")

TRAIN_RATIO = 0.8

# Số partition shuffle.
# Nếu máy bị thiếu RAM hoặc ổ đĩa chậm, có thể tăng lên 24 hoặc 32.
SHUFFLE_PARTITIONS = 200


# ============================================================
# SPARK SESSION
# ============================================================

def create_spark():
    Path(SPARK_TMP).mkdir(parents=True, exist_ok=True)

    spark = (
        SparkSession.builder
        .master("local[4]")
        .appName("Amazon-Temporal-Train-Test-Split")
        .config("spark.local.dir", SPARK_TMP)
        .config("spark.driver.memory", "4g")
        .config("spark.driver.maxResultSize", "512m")
        .config(
            "spark.default.parallelism",
            str(SHUFFLE_PARTITIONS)
        )
        .config(
            "spark.sql.shuffle.partitions",
            str(SHUFFLE_PARTITIONS)
        )
        .config("spark.sql.adaptive.enabled", "true")
        .config(
            "spark.sql.adaptive.coalescePartitions.enabled",
            "true"
        )
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    return spark


# ============================================================
# WRITE EXACTLY ONE PHYSICAL PARQUET FILE
# ============================================================

def save_single_parquet(df, output_file: Path):
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    tmp_dir = output_file.parent / f".{output_file.stem}_tmp"

    # Xóa đầu ra cũ nếu có
    if output_file.exists():
        if output_file.is_dir():
            shutil.rmtree(output_file)
        else:
            output_file.unlink()

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    try:
        (
            df.coalesce(1)
            .write
            .mode("overwrite")
            .option("compression",  "uncompressed")
            .parquet(str(tmp_dir))
        )

        parquet_files = list(tmp_dir.glob("*.parquet"))

        if len(parquet_files) != 1:
            raise RuntimeError(
                f"Expected exactly 1 Parquet file, "
                f"found {len(parquet_files)} in {tmp_dir}"
            )

        # Di chuyển file Parquet ra vị trí đích
        shutil.move(
            str(parquet_files[0]),
            str(output_file)
        )

        print(f"Saved: {output_file}")
        print(
            f"File size: "
            f"{output_file.stat().st_size / (1024 ** 3):.2f} GiB"
        )

    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ============================================================
# MAIN
# ============================================================

def main():
    if not 0 < TRAIN_RATIO < 1:
        raise ValueError("TRAIN_RATIO must be between 0 and 1.")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    spark = create_spark()

    try:
        # ----------------------------------------------------
        # 1. READ INTERACTIONS
        # ----------------------------------------------------

        print("Reading interactions...")

        interactions = (
            spark.read.parquet(str(INPUT_FILE))
            .select(
                F.col("user_idx").cast("int").alias("user_idx"),
                F.col("item_idx").cast("int").alias("item_idx"),
                F.col("rating").cast("float").alias("rating"),
                F.col("unixReviewTime")
                .cast("long")
                .alias("unixReviewTime"),
            )
            .filter(
                F.col("user_idx").isNotNull()
                & F.col("item_idx").isNotNull()
                & F.col("rating").isNotNull()
                & F.col("unixReviewTime").isNotNull()
            )
        )

        # ----------------------------------------------------
        # 2. REMOVE DUPLICATE USER-ITEM PAIRS
        # ----------------------------------------------------
        # Một user chỉ có một tương tác với mỗi item.
        # Giữ lại rating mới nhất theo unixReviewTime.
        #
        # Nếu dữ liệu đầu vào đã được deduplicate theo quy tắc
        # này, bước này vẫn bảo đảm split không bị trùng cặp.

        print("Removing duplicate user-item interactions...")

        w_latest = (
            Window
            .partitionBy("user_idx", "item_idx")
            .orderBy(F.col("unixReviewTime").desc())
        )

        interactions = (
            interactions
            .withColumn("_duplicate_rank", F.row_number().over(w_latest))
            .filter(F.col("_duplicate_rank") == 1)
            .drop("_duplicate_rank")
        )

        # ----------------------------------------------------
        # 3. ORDER INTERACTIONS BY TIME PER USER
        # ----------------------------------------------------

        print("Ordering interactions by user and timestamp...")

        w_order = (
            Window
            .partitionBy("user_idx")
            .orderBy(
                F.col("unixReviewTime").asc(),
                F.col("item_idx").asc()
            )
        )

        w_user = Window.partitionBy("user_idx")

        ranked = (
            interactions
            .withColumn("_rn", F.row_number().over(w_order))
            .withColumn("_n", F.count("*").over(w_user))
            # Phải có ít nhất 2 item để tạo train và test
            .filter(F.col("_n") >= 2)
        )

        # ----------------------------------------------------
        # 4. COMPUTE TRAIN SIZE FOR EACH USER
        # ----------------------------------------------------
        # Ví dụ:
        # n = 5  -> train = 4, test = 1
        # n = 10 -> train = 8, test = 2
        #
        # Làm tròn xuống theo floor; bảo đảm train ít nhất 1
        # và test ít nhất 1.

        ranked = ranked.withColumn(
            "_train_n",
            F.greatest(
                F.lit(1),
                F.least(
                    F.col("_n") - F.lit(1),
                    F.floor(
                        F.col("_n") * F.lit(TRAIN_RATIO)
                    ).cast("long")
                )
            )
        )

        # ----------------------------------------------------
        # 5. CREATE TRAIN AND TEST
        # ----------------------------------------------------

        columns = [
            "user_idx",
            "item_idx",
            "rating",
            "unixReviewTime",
        ]

        train = (
            ranked
            .filter(F.col("_rn") <= F.col("_train_n"))
            .select(*columns)
        )

        test = (
            ranked
            .filter(F.col("_rn") > F.col("_train_n"))
            .select(*columns)
        )

        # ----------------------------------------------------
        # 6. WRITE OUTPUT FILES
        # ----------------------------------------------------

        print("Writing train.parquet...")
        save_single_parquet(
            train,
            OUTPUT_DIR / "train.parquet"
        )

        # Giải phóng tham chiếu đến train trước khi ghi test
        train.unpersist(blocking=False)
        del train

        print("Writing test.parquet...")
        save_single_parquet(
            test,
            OUTPUT_DIR / "test.parquet"
        )

        test.unpersist(blocking=False)
        del test

        print("\nSplit completed.")
        print(f"Output directory: {OUTPUT_DIR}")
        print(f"Train ratio target: {TRAIN_RATIO:.0%}")
        print(f"Test ratio target: {1 - TRAIN_RATIO:.0%}")
        print("Split strategy: chronological per user")
        print("Validation set: not created")

    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise

    finally:
        spark.stop()


if __name__ == "__main__":
    main()