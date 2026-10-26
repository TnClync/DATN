from pathlib import Path
import shutil
import gc
import tempfile

from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import StringIndexer


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
APP_NAME = "AmazonCF"
SPARK_MASTER = "local[6]"

INPUT_FILE = str(PROJECT_DIR / "data" / "Books_5.json" / "Books_5.json")

OUTPUT_DIR = PROJECT_DIR / "data"

SPARK_TMP = str(Path(tempfile.gettempdir()) / "datn-spark")


# ============================================================
# RAW SCHEMA
# ============================================================

RAW_SCHEMA = """
    asin STRING,
    overall DOUBLE,
    reviewText STRING,
    reviewTime STRING,
    reviewerID STRING,
    reviewerName STRING,
    style STRING,
    summary STRING,
    unixReviewTime LONG,
    verified BOOLEAN
"""


# ============================================================
# SPARK
# ============================================================

def create_spark():

    Path(SPARK_TMP).mkdir(
        parents=True,
        exist_ok=True
    )

    spark = (
        SparkSession.builder
        .master(SPARK_MASTER)
        .appName(APP_NAME)

        # Spark temporary / shuffle files
        .config(
            "spark.local.dir",
            SPARK_TMP
        )

        # Memory
        .config(
            "spark.driver.memory",
            "5g"
        )
        .config(
            "spark.driver.maxResultSize",
            "512m"
        )

        # Parallelism
        .config(
            "spark.default.parallelism",
            "16"
        )
        .config(
            "spark.sql.shuffle.partitions",
            "16"
        )

        # AQE
        .config(
            "spark.sql.adaptive.enabled",
            "true"
        )
        .config(
            "spark.sql.adaptive.coalescePartitions.enabled",
            "true"
        )

        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    return spark


# ============================================================
# SAVE SINGLE PARQUET FILE
# ============================================================

def save_single_parquet(
    df,
    output_file
):

    output_file = Path(output_file)

    tmp_dir = (
        output_file.parent
        / f".{output_file.stem}_tmp"
    )

    # Remove old output
    if output_file.exists():

        if output_file.is_dir():
            shutil.rmtree(output_file)

        else:
            output_file.unlink()

    # Remove old temporary directory
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Force exactly one physical parquet file
    (
        df
        .coalesce(1)
        .write
        .mode("overwrite")
        .option(
            "compression",
            "uncompressed"
        )
        .parquet(str(tmp_dir))
    )

    parquet_files = list(
        tmp_dir.glob("*.parquet")
    )

    if len(parquet_files) != 1:

        raise RuntimeError(
            f"Expected exactly 1 parquet file, "
            f"found {len(parquet_files)}"
        )

    shutil.move(
        str(parquet_files[0]),
        str(output_file)
    )

    shutil.rmtree(tmp_dir)


# ============================================================
# LOAD RAW DATA
# ============================================================

def load_raw(spark):

    return (
        spark.read
        .schema(RAW_SCHEMA)
        .json(INPUT_FILE)

        .select(
            "reviewerID",
            "asin",
            "overall",
            "reviewText",
            "summary",
            "unixReviewTime"
        )

        .filter(
            F.col("reviewerID").isNotNull()
            &
            F.col("asin").isNotNull()
            &
            F.col("overall").isNotNull()
        )
    )


# ============================================================
# DEDUPLICATE USER-ITEM
# ============================================================

def deduplicate(df):

    latest = (
        df
        .groupBy(
            "reviewerID",
            "asin"
        )
        .agg(
            F.max(
                "unixReviewTime"
            ).alias(
                "unixReviewTime"
            )
        )
    )

    result = (
        latest

        .join(
            df,
            on=[
                "reviewerID",
                "asin",
                "unixReviewTime"
            ],
            how="inner"
        )

        .select(
            "reviewerID",
            "asin",
            "overall",
            "reviewText",
            "summary",
            "unixReviewTime"
        )

        .dropDuplicates(
            [
                "reviewerID",
                "asin"
            ]
        )
    )

    return result


# ============================================================
# USER MAPPING
# reviewerID -> user_idx
# ============================================================

def create_user_mapping(df):

    users = (
        df
        .select("reviewerID")
        .distinct()
    )

    indexer = StringIndexer(
        inputCol="reviewerID",
        outputCol="user_idx",
        handleInvalid="error"
    )

    model = indexer.fit(users)

    return (
        model
        .transform(users)

        .select(
            "reviewerID",

            F.col("user_idx")
            .cast("int")
            .alias("user_idx")
        )
    )


# ============================================================
# ITEM MAPPING
# asin -> item_idx
# ============================================================

def create_item_mapping(df):

    items = (
        df
        .select("asin")
        .distinct()
    )

    indexer = StringIndexer(
        inputCol="asin",
        outputCol="item_idx",
        handleInvalid="error"
    )

    model = indexer.fit(items)

    return (
        model
        .transform(items)

        .select(
            "asin",

            F.col("item_idx")
            .cast("int")
            .alias("item_idx")
        )
    )


# ============================================================
# INTERACTIONS
#
# FINAL:
# user_idx | item_idx | rating | unixReviewTime
# ============================================================

def create_interactions(
    df,
    user_mapping,
    item_mapping
):

    return (
        df

        .join(
            user_mapping,
            on="reviewerID",
            how="inner"
        )

        .join(
            item_mapping,
            on="asin",
            how="inner"
        )

        .select(
            "user_idx",
            "item_idx",

            F.col("overall")
            .cast("float")
            .alias("rating"),

            F.col("unixReviewTime")
            .cast("long")
            .alias("unixReviewTime")
        )
    )


# ============================================================
# PRODUCT TEXT
#
# FINAL:
# item_idx | reviewText | summary
# ============================================================

def create_product_text(
    df,
    item_mapping
):

    return (
        df

        .join(
            item_mapping,
            on="asin",
            how="inner"
        )

        .select(
            "item_idx",
            "reviewText",
            "summary"
        )

        .filter(
            F.col("reviewText").isNotNull()
            |
            F.col("summary").isNotNull()
        )

        .dropDuplicates(
            ["item_idx"]
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    spark = create_spark()

    try:

        # ----------------------------------------------------
        # 1. LOAD RAW
        # ----------------------------------------------------

        print("\n[1/7] Loading raw data...")

        df = load_raw(spark)


        # ----------------------------------------------------
        # 2. DEDUPLICATE
        # ----------------------------------------------------

        print("[2/7] Deduplicating user-item...")

        df = deduplicate(df)


        # ----------------------------------------------------
        # 3. USER MAPPING
        # ----------------------------------------------------

        print("[3/7] Creating user mapping...")

        user_mapping = create_user_mapping(df)

        save_single_parquet(
            user_mapping,
            OUTPUT_DIR / "user_mapping.parquet"
        )

        del user_mapping
        gc.collect()

        spark.catalog.clearCache()


        # ----------------------------------------------------
        # 4. ITEM MAPPING
        # ----------------------------------------------------

        print("[4/7] Creating item mapping...")

        item_mapping = create_item_mapping(df)

        save_single_parquet(
            item_mapping,
            OUTPUT_DIR / "item_mapping.parquet"
        )

        # Keep item_mapping because it is needed
        # for interactions and product_text.


        # ----------------------------------------------------
        # 5. RELOAD USER MAPPING
        # ----------------------------------------------------

        print("[5/7] Loading saved user mapping...")

        user_mapping = (
            spark.read
            .parquet(
                str(
                    OUTPUT_DIR
                    / "user_mapping.parquet"
                )
            )
        )


        # ----------------------------------------------------
        # 6. INTERACTIONS
        # ----------------------------------------------------

        print("[6/7] Creating interactions...")

        interactions = create_interactions(
            df,
            user_mapping,
            item_mapping
        )

        save_single_parquet(
            interactions,
            OUTPUT_DIR / "interactions.parquet"
        )

        del interactions
        gc.collect()

        spark.catalog.clearCache()


        # ----------------------------------------------------
        # 7. PRODUCT TEXT
        # ----------------------------------------------------

        print("[7/7] Creating product text...")

        product_text = create_product_text(
            df,
            item_mapping
        )

        save_single_parquet(
            product_text,
            OUTPUT_DIR / "product_text.parquet"
        )

        del product_text
        gc.collect()

        spark.catalog.clearCache()


        # ----------------------------------------------------
        # CLEANUP
        # ----------------------------------------------------

        del df
        del user_mapping
        del item_mapping

        gc.collect()

        spark.catalog.clearCache()

        print("\n========================================")
        print("DATA PREPROCESSING COMPLETED")
        print("========================================")
        print(
            f"Output: {OUTPUT_DIR}"
        )

    finally:

        spark.stop()

        gc.collect()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()