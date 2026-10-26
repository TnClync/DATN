
import json
import tempfile
from pathlib import Path

from pyspark.sql import SparkSession, functions as F

PROJECT_DIR = Path(__file__).resolve().parents[1]
SPARK_TMP = str(Path(tempfile.gettempdir()) / "datn-spark")


class MostPopRecommender:
    def __init__(
        self,
        train_path=str(PROJECT_DIR / "data" / "splits" / "train.parquet"),
        json_path=str(PROJECT_DIR / "outputs" / "mostpop.json"),
        top_n=1000,
        positive_threshold=4.0,
    ):
        self.train_path = train_path
        self.json_path = Path(json_path)
        self.top_n = top_n
        self.positive_threshold = positive_threshold

        self.json_path.parent.mkdir(parents=True, exist_ok=True)

        self.spark = (
            SparkSession.builder
            .appName("DATN_MostPop")
            .master("local[4]")
            .config("spark.local.dir", SPARK_TMP)
            .config("spark.sql.shuffle.partitions", "16")
            .getOrCreate()
        )
        self.spark.sparkContext.setLogLevel("WARN")

        if self.json_path.exists():
            self.load()
        else:
            self.fit()

    def fit(self):
        """Train MostPop và lưu thành một file JSON."""
        df = self.spark.read.parquet(self.train_path)

        required = {"item_idx", "rating"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Thiếu các cột bắt buộc: {sorted(missing)}"
            )

        positive = df.filter(
            F.col("item_idx").isNotNull()
            & F.col("rating").isNotNull()
            & (F.col("rating") >= self.positive_threshold)
        )

        aggregations = [
            F.count("*").alias("positive_interactions")
        ]

        if "user_idx" in positive.columns:
            aggregations.append(
                F.countDistinct("user_idx").alias("unique_users")
            )

        popularity = (
            positive.groupBy("item_idx")
            .agg(*aggregations)
            .orderBy(
                F.desc("positive_interactions"),
                *(
                    [F.desc("unique_users")]
                    if "unique_users" in positive.columns
                    else []
                ),
                F.asc("item_idx"),
            )
            .limit(self.top_n)
        )

        # Chỉ thu thập Top-N, không collect toàn bộ interactions.
        rows = [row.asDict() for row in popularity.collect()]

        payload = {
            "top_n": self.top_n,
            "positive_threshold": self.positive_threshold,
            "items": rows,
        }

        # Ghi file tạm rồi đổi tên để tránh JSON dở dang.
        temp_path = self.json_path.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        temp_path.replace(self.json_path)

        self.items = rows
        return self

    def load(self):
        """Load model từ JSON đã lưu."""
        with self.json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        self.items = payload["items"]
        return self

    def recommend(self, n=10):
        """Trả về Top-N item phổ biến dưới dạng list[dict]."""
        return self.items[:n]

    def close(self):
        """Đóng SparkSession khi không còn sử dụng."""
        self.spark.stop()
