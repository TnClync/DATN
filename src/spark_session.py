"""Spark session helper with a dedicated temp directory (laptop-friendly)."""
import os
import tempfile
from pathlib import Path

from pyspark.sql import SparkSession


def get_spark(app_name: str = "DATN-Recommender", temp_dir: str | None = None, cores: int = 6, memory_gb: int = 4):
    """Create a SparkSession constrained to laptop resources and a temp dir."""
    if temp_dir is None:
        temp_dir = str(Path(tempfile.gettempdir()) / "datn_spark")
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    os.environ["SPARK_LOCAL_DIRS"] = temp_dir
    spark = (
        SparkSession.builder.appName(app_name)
        .master(f"local[{cores}]")
        .config("spark.driver.memory", f"{memory_gb}g")
        .config("spark.executor.memory", f"{memory_gb}g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.local.dir", temp_dir)
        .config("spark.driver.extraJavaOptions", f"-Djava.io.tmpdir={temp_dir}")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark, temp_dir
