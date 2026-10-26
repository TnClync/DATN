
from pathlib import Path
import shutil
import gc
import sys
import tempfile

from pyspark.sql import SparkSession, functions as F
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator


# ============================================================
# 1. CONFIG
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_DIR / "data" / "splits"

OUTPUT_DIR = PROJECT_DIR / "models" / "als"
SPARK_TMP = str(Path(tempfile.gettempdir()) / "datn-spark")

TRAIN_FILE = SPLIT_DIR / "train.parquet"
TEST_FILE = SPLIT_DIR / "test.parquet"

MODEL_DIR = OUTPUT_DIR / "als_model"
USER_FACTORS_FILE = OUTPUT_DIR / "user_factors.parquet"
ITEM_FACTORS_FILE = OUTPUT_DIR / "item_factors.parquet"

# Initial configuration for a laptop with 8 GB RAM.
RANK = 50
MAX_ITER = 20
REG_PARAM = 0.1
SEED = 42

SHUFFLE_PARTITIONS = 2000


# ============================================================
# 2. SPARK SESSION
# ============================================================

def create_spark():
    Path(SPARK_TMP).mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    spark = (
        SparkSession.builder
        .master("local[6]")
        .appName("Amazon-ALS-Train-Test")
        .config("spark.local.dir", SPARK_TMP)
        .config("spark.master", "local[6]")
        .config("spark.driver.memory", "3g")
        .config("spark.driver.maxResultSize", "512m")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.default.parallelism", "16")
        .config("spark.sql.adaptive.enabled", "true")
        .config(
            "spark.sql.adaptive.coalescePartitions.enabled",
            "true"
        )

        # Hạn chế giữ dữ liệu trung gian trong bộ nhớ
        .config("spark.memory.fraction", "0.6")
        .config("spark.memory.storageFraction", "0.3")

        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    return spark


# ============================================================
# 3. SAVE ONE PHYSICAL PARQUET FILE
# ============================================================

def save_single_parquet(df, output_file):
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    tmp_dir = output_file.parent / f".{output_file.stem}_tmp"

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
            .option("compression", "snappy")
            .parquet(str(tmp_dir))
        )

        parquet_files = list(tmp_dir.glob("*.parquet"))

        if len(parquet_files) != 1:
            raise RuntimeError(
                f"Expected one Parquet file, "
                f"found {len(parquet_files)}: {tmp_dir}"
            )

        shutil.move(
            str(parquet_files[0]),
            str(output_file)
        )

        print(f"Saved: {output_file}")
        print(
            f"Size: "
            f"{output_file.stat().st_size / (1024 ** 3):.2f} GiB"
        )

    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ============================================================
# 4. LOAD AND VALIDATE INPUT DATA
# ============================================================

def load_split(spark, file_path):
    if not file_path.exists():
        raise FileNotFoundError(
            f"Split file not found: {file_path}"
        )

    df = (
        spark.read.parquet(str(file_path))
        .select(
            F.col("user_idx").cast("int").alias("user_idx"),
            F.col("item_idx").cast("int").alias("item_idx"),
            F.col("rating").cast("float").alias("rating")
        )
        .filter(
            F.col("user_idx").isNotNull()
            & F.col("item_idx").isNotNull()
            & F.col("rating").isNotNull()
            & ~F.isnan("rating")
            & (F.col("rating") >= 1.0)
            & (F.col("rating") <= 5.0)
        )
    )

    return df


# ============================================================
# 5. TRAIN ALS
# ============================================================

def train_als(train):
    als = ALS(
        userCol="user_idx",
        itemCol="item_idx",
        ratingCol="rating",
        rank=RANK,
        maxIter=MAX_ITER,
        regParam=REG_PARAM,
        implicitPrefs=False,
        nonnegative=Falseu,
        coldStartStrategy="drop",
        seed=SEED,
        checkpointInterval=10
    )

    return als.fit(train)


# ============================================================
# 6. EVALUATE RMSE AND MAE
# ============================================================

def evaluate(model, dataset, split_name):
    predictions = (
        model.transform(dataset)
        .select("rating", "prediction")
        .filter(
            F.col("prediction").isNotNull()
            & ~F.isnan("prediction")
            & F.col("rating").isNotNull()
        )
    )

    evaluator = RegressionEvaluator(
        labelCol="rating",
        predictionCol="prediction",
        metricName="rmse"
    )

    rmse = evaluator.evaluate(predictions)

    evaluator.setMetricName("mae")
    mae = evaluator.evaluate(predictions)

    print(f"\n===== {split_name.upper()} METRICS =====")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE : {mae:.6f}")

    del predictions
    gc.collect()

    return {
        "rmse": rmse,
        "mae": mae
    }


# ============================================================
# 7. SAVE MODEL AND FACTORS
# ============================================================

def save_model_and_factors(model):
    # Remove previous model artifacts.
    if MODEL_DIR.exists():
        if MODEL_DIR.is_dir():
            shutil.rmtree(MODEL_DIR)
        else:
            MODEL_DIR.unlink()

    # Save native Spark ALS model.
    model.write().save(str(MODEL_DIR))

    # Factor schemas:
    # userFactors: id, features
    # itemFactors: id, features
    #
    # The id corresponds to user_idx / item_idx.

    save_single_parquet(
        model.userFactors,
        USER_FACTORS_FILE
    )

    save_single_parquet(
        model.itemFactors,
        ITEM_FACTORS_FILE
    )

    print("\n===== SAVED ARTIFACTS =====")
    print(f"Model        : {MODEL_DIR}")
    print(f"User factors : {USER_FACTORS_FILE}")
    print(f"Item factors : {ITEM_FACTORS_FILE}")


# ============================================================
# 8. MAIN
# ============================================================

def main():
    spark = create_spark()
    model = None
    train = None
    test = None

    try:
        # ----------------------------------------------------
        # STEP 1: LOAD PRE-SPLIT DATA
        # ----------------------------------------------------

        print("[1/5] Loading train split...")
        train = load_split(spark, TRAIN_FILE)

        print("[2/5] Loading test split...")
        test = load_split(spark, TEST_FILE)

        # ----------------------------------------------------
        # STEP 2: TRAIN
        # ----------------------------------------------------

        print("[3/5] Training ALS...")
        print(f"rank       : {RANK}")
        print(f"maxIter    : {MAX_ITER}")
        print(f"regParam   : {REG_PARAM}")
        print(f"partitions : {SHUFFLE_PARTITIONS}")

        model = train_als(train)

        # ----------------------------------------------------
        # STEP 3: EVALUATE ON TEST ONLY
        # ----------------------------------------------------

        print("[4/5] Evaluating on test...")
        test_metrics = evaluate(
            model,
            test,
            "test"
        )

        # ----------------------------------------------------
        # STEP 4: SAVE MODEL AND FACTORS
        # ----------------------------------------------------

        print("[5/5] Saving model and factors...")
        save_model_and_factors(model)

        # ----------------------------------------------------
        # STEP 5: SUMMARY
        # ----------------------------------------------------

        print("\n===== TRAINING SUMMARY =====")
        print(f"rank       : {RANK}")
        print(f"maxIter    : {MAX_ITER}")
        print(f"regParam   : {REG_PARAM}")
        print(f"seed       : {SEED}")
        print(f"test RMSE  : {test_metrics['rmse']:.6f}")
        print(f"test MAE   : {test_metrics['mae']:.6f}")

    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise

    finally:
        # Do not persist the full train/test datasets.
        # Release Python references before stopping Spark.
        model = None
        train = None
        test = None

        gc.collect()
        spark.stop()


if __name__ == "__main__":
    main()