
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


class CollaborativeRecommender:
    """
    Collaborative Filtering recommender backed by exported Spark ALS factors.

    Expected files:
        userFactors.parquet
        itemFactors.parquet
        user_mapping.parquet
        item_mapping.parquet
        interactions.parquet

    Returns:
        List of (asin, score) tuples.
    """

    def __init__(self, spark, model_dir="als_output"):
        self.spark = spark
        self.model_dir = model_dir.rstrip("/")

        # Read distributed Parquet data; do not collect all factors.
        self.user_factors = (
            spark.read.parquet(
                f"{self.model_dir}/userFactors.parquet"
            )
            .select(
                F.col("id").cast("long").alias("user_idx"),
                F.col("features").alias("user_features"),
            )
        )

        self.item_factors = (
            spark.read.parquet(
                f"{self.model_dir}/itemFactors.parquet"
            )
            .select(
                F.col("id").cast("long").alias("item_idx"),
                F.col("features").alias("item_features"),
            )
        )

        self.user_mapping = spark.read.parquet(
            f"{self.model_dir}/user_mapping.parquet"
        ).select("reviewerID", "user_idx")

        self.item_mapping = spark.read.parquet(
            f"{self.model_dir}/item_mapping.parquet"
        ).select("item_idx", "item_key")

        self.interactions = spark.read.parquet(
            f"{self.model_dir}/interactions.parquet"
        ).select("user_idx", "item_idx")

    def recommend(
        self,
        user_raw_id,
        N=10,
        domain=None,
        exclude_seen=True,
    ):
        """
        domain: None, "Books", or "Electronics".
        Excludes every previously interacted item by default.
        """

        if N <= 0:
            return []

        # 1. Map raw reviewerID to model user_idx.
        user_row = (
            self.user_mapping
            .filter(F.col("reviewerID") == user_raw_id)
            .select("user_idx")
            .limit(1)
            .collect()
        )

        if not user_row:
            return []

        user_idx = int(user_row[0]["user_idx"])

        # 2. Load this user's latent factor only.
        factor_row = (
            self.user_factors
            .filter(F.col("user_idx") == user_idx)
            .select("user_features")
            .limit(1)
            .collect()
        )

        if not factor_row or factor_row[0]["user_features"] is None:
            return []

        user_vector = factor_row[0]["user_features"]

        # 3. Calculate dot product against all item factors.
        # The item factors remain distributed in Spark.
        user_array = F.array(
            *[F.lit(float(value)) for value in user_vector]
        )

        candidates = (
            self.item_factors
            .withColumn("_user_features", user_array)
            .withColumn(
                "score",
                F.aggregate(
                    F.zip_with(
                        F.col("_user_features"),
                        F.col("item_features"),
                        lambda u, i: u * i,
                    ),
                    F.lit(0.0),
                    lambda acc, value: acc + value,
                ),
            )
            .select("item_idx", "score")
            .join(self.item_mapping, on="item_idx", how="inner")
        )

        # 4. Decode the domain and original ASIN.
        candidates = (
            candidates
            .withColumn(
                "domain",
                F.element_at(
                    F.split(F.col("item_key"), "::"), 1
                ),
            )
            .withColumn(
                "asin",
                F.element_at(
                    F.split(F.col("item_key"), "::"), 2
                ),
            )
        )

        if domain is not None:
            candidates = candidates.filter(
                F.col("domain") == domain
            )

        # 5. Exclude items the user has interacted with.
        if exclude_seen:
            seen = (
                self.interactions
                .filter(F.col("user_idx") == user_idx)
                .select("item_idx")
                .distinct()
            )

            candidates = candidates.join(
                seen,
                on="item_idx",
                how="left_anti",
            )

        # 6. Return top-N ranked recommendations.
        rows = (
            candidates
            .filter(F.col("asin").isNotNull())
            .orderBy(F.col("score").desc())
            .select("asin", "score")
            .limit(N)
            .collect()
        )

        return [
            (row["asin"], float(row["score"]))
            for row in rows
        ]

