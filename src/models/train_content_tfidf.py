
import tempfile
from pathlib import Path

from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, CountVectorizer, IDF


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]
ITEM_PROFILE_PATH = str(PROJECT_DIR / "data" / "Item_profile")
OUTPUT_DIR = str(PROJECT_DIR / "models" / "content_based" / "tfidf_model")
VECTORS_DIR = str(PROJECT_DIR / "models" / "content_based" / "item_vectors")
SPARK_LOCAL_DIR = str(Path(tempfile.gettempdir()) / "datn-spark")

VOCAB_SIZE = 200_000
MIN_DF = 2
SHUFFLE_PARTITIONS = 64


# ============================================================
# SPARK
# ============================================================

spark = (
    SparkSession.builder
    .appName("DATN_ContentBased_TFIDF")
    .master("local[4]")
    .config("spark.local.dir", SPARK_LOCAL_DIR)
    .config("spark.sql.shuffle.partitions", str(SHUFFLE_PARTITIONS))
    .config("spark.sql.parquet.compression.codec", "zstd")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


def main():
    source = Path(ITEM_PROFILE_PATH)

    if not source.exists():
        raise FileNotFoundError(
            f"Item profile not found: {source}"
        )

    # --------------------------------------------------------
    # 1. Read item metadata
    # --------------------------------------------------------
    df = spark.read.parquet(ITEM_PROFILE_PATH)

    print("Schema:")
    df.printSchema()

    available = set(df.columns)

    if "item_idx" not in available:
        raise ValueError("Item_profile thiếu item_idx.")

    fields = [
        name
        for name in [
            "title",
            "brand",
            "category",
            "feature",
            "description",
        ]
        if name in available
    ]

    if not fields:
        raise ValueError(
            "Không tìm thấy title/brand/category/feature/description."
        )

    print("Text fields:", fields)

    # --------------------------------------------------------
    # 2. Build content text
    # --------------------------------------------------------
    # Cast to string supports both scalar and array metadata.
    text_columns = [
        F.coalesce(
            F.col(name).cast("string"),
            F.lit(""),
        )
        for name in fields
    ]

    df = (
        df.select(
            *[
                F.col(c)
                for c in [
                    "item_idx",
                    "asin",
                    "title",
                    "brand",
                    "category",
                    "price",
                ]
                if c in available
            ],
            F.concat_ws(" ", *text_columns).alias("content_text"),
        )
        .filter(F.col("content_text").isNotNull())
        .filter(F.length(F.trim("content_text")) > 0)
        .dropDuplicates(["item_idx"])
    )

    print("Training item count:", df.count())

    # --------------------------------------------------------
    # 3. Tokenize text
    # --------------------------------------------------------
    tokenizer = RegexTokenizer(
        inputCol="content_text",
        outputCol="tokens",
        pattern=r"\W+",
        minTokenLength=2,
        toLowercase=True,
    )

    # --------------------------------------------------------
    # 4. Learn vocabulary
    # --------------------------------------------------------
    count_vectorizer = CountVectorizer(
        inputCol="tokens",
        outputCol="tf",
        vocabSize=VOCAB_SIZE,
        minDF=MIN_DF,
        binary=False,
    )

    # --------------------------------------------------------
    # 5. Compute IDF
    # --------------------------------------------------------
    idf = IDF(
        inputCol="tf",
        outputCol="tfidf",
        minDocFreq=MIN_DF,
    )

    pipeline = Pipeline(
        stages=[
            tokenizer,
            count_vectorizer,
            idf,
        ]
    )

    # --------------------------------------------------------
    # 6. Train
    # --------------------------------------------------------
    print("\nTraining TF-IDF pipeline...")
    model = pipeline.fit(df)

    # --------------------------------------------------------
    # 7. Save trained model
    # --------------------------------------------------------
    Path(OUTPUT_DIR).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.write().overwrite().save(OUTPUT_DIR)

    print("Saved TF-IDF model:", OUTPUT_DIR)

    # --------------------------------------------------------
    # 8. Transform product metadata
    # --------------------------------------------------------
    # Keep vectors distributed in Parquet.
    # Do not collect millions of vectors into Python RAM.
    print("\nTransforming product metadata...")

    transformed = model.transform(df)

    output_columns = [
        c
        for c in [
            "item_idx",
            "asin",
            "title",
            "brand",
            "category",
            "price",
            "tfidf",
        ]
        if c in transformed.columns
    ]

    (
        transformed
        .select(*output_columns)
        .write
        .mode("overwrite")
        .parquet(VECTORS_DIR)
    )

    print("Saved product vectors:", VECTORS_DIR)

    # --------------------------------------------------------
    # 9. Report actual artifacts
    # --------------------------------------------------------
    cv_model = model.stages[1]
    print("Learned vocabulary size:", len(cv_model.vocabulary))
    print("TF-IDF training completed.")

    spark.stop()


if __name__ == "__main__":
    main()
