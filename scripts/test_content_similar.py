
import argparse
import os
import tempfile
from pathlib import Path

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import DoubleType


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
VECTORS_DIR = str(PROJECT_DIR / "models" / "content_based" / "item_vectors")
SPARK_LOCAL_DIR = str(Path(tempfile.gettempdir()) / "datn-spark")

SHUFFLE_PARTITIONS = 2


# ============================================================
# SPARK
# ============================================================

def build_spark():
    os.makedirs(SPARK_LOCAL_DIR, exist_ok=True)

    return (
        SparkSession.builder
        .appName("AmazonTFIDFSimilarItems")
        .master("local[4]")
        .config("spark.local.dir", SPARK_LOCAL_DIR)
        .config(
            "spark.sql.shuffle.partitions",
            str(SHUFFLE_PARTITIONS),
        )
        .config("spark.sql.parquet.compression.codec", "zstd")
        .getOrCreate()
    )


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(query_vector, item_vector):
    """
    Tính cosine similarity giữa hai vector TF-IDF.

    Công thức:
        cosine(A, B) = dot(A, B) / (||A|| * ||B||)

    Tính trực tiếp để không phụ thuộc giả định rằng vector
    đã được chuẩn hóa L2.
    """
    if query_vector is None or item_vector is None:
        return 0.0

    if query_vector.size != item_vector.size:
        return 0.0

    dot_product = float(query_vector.dot(item_vector))

    query_norm = float(query_vector.norm(2))
    item_norm = float(item_vector.norm(2))

    denominator = query_norm * item_norm

    if denominator == 0:
        return 0.0

    return dot_product / denominator


# ============================================================
# RECOMMEND SIMILAR ITEMS
# ============================================================

def recommend_similar_items(spark, asin, top_n=10):
    """
    Parameters
    ----------
    spark:
        SparkSession đang hoạt động.

    asin:
        ASIN của sản phẩm gốc.

    top_n:
        Số sản phẩm tương tự cần trả về.

    Returns
    -------
    list[dict]:
        Danh sách sản phẩm kèm cosine similarity.
    """

    if top_n <= 0:
        raise ValueError("top_n must be greater than zero.")

    if not os.path.exists(VECTORS_DIR):
        raise FileNotFoundError(
            f"Vector directory does not exist: {VECTORS_DIR}"
        )

    print("Loading product vectors...")

    items = spark.read.parquet(VECTORS_DIR)

    # Actual schema reported in your Spark error:
    # item_idx, asin, title, brand, category, price, tfidf
    required_columns = {"asin", "tfidf"}
    missing_columns = required_columns - set(items.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}. "
            f"Actual schema: {items.columns}"
        )

    print("Looking up source product...")

    query_rows = (
        items
        .filter(F.col("asin") == asin)
        .select("asin", "tfidf")
        .limit(1)
        .collect()
    )

    if not query_rows:
        print(f"Product not found: {asin}")
        return []

    query_vector = query_rows[0]["tfidf"]

    if query_vector is None:
        print(f"Product has no TF-IDF vector: {asin}")
        return []

    print(f"Source product: {asin}")
    print("Computing cosine similarity...")

    similarity_udf = F.udf(
        lambda vector: cosine_similarity(query_vector, vector),
        DoubleType(),
    )

    # Exclude the source product itself.
    scored = (
        items
        .filter(F.col("asin") != asin)
        .filter(F.col("tfidf").isNotNull())
        .withColumn(
            "similarity",
            similarity_udf(F.col("tfidf")),
        )
        .filter(F.col("similarity") > 0)
        .orderBy(F.desc("similarity"))
        .limit(top_n)
    )

    display_columns = [
        column
        for column in [
            "asin",
            "title",
            "brand",
            "category",
            "price",
            "similarity",
        ]
        if column in scored.columns
    ]

    results = [
        row.asDict(recursive=True)
        for row in scored.select(*display_columns).collect()
    ]

    return results


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(results):
    if not results:
        print("\nNo similar products found.")
        return

    print(f"\nFound {len(results)} similar products")
    print("=" * 80)

    for rank, product in enumerate(results, start=1):
        print(f"\nRank: {rank}")
        print(f"ASIN: {product.get('asin')}")
        print(f"Title: {product.get('title')}")
        print(f"Brand: {product.get('brand')}")
        print(f"Category: {product.get('category')}")
        print(f"Price: {product.get('price')}")
        print(
            "Cosine similarity: "
            f"{product.get('similarity', 0.0):.6f}"
        )

    print("\n" + "=" * 80)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Recommend similar products using TF-IDF."
    )

    parser.add_argument(
        "--asin",
        type=str,
        required=True,
        help="ASIN of the source product.",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of similar products to return.",
    )

    args = parser.parse_args()

    spark = build_spark()
    spark.sparkContext.setLogLevel("ERROR")

    try:
        results = recommend_similar_items(
            spark=spark,
            asin=args.asin,
            top_n=args.top_n,
        )

        print_results(results)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
