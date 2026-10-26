from pyspark.sql import SparkSession

from config import (
    APP_NAME,
    MASTER,
    SHUFFLE_PARTITIONS,
    SPARK_DRIVER_MEMORY,
    SPARK_EXECUTOR_MEMORY,
    SPARK_MEMORY_FRACTION,
    SPARK_MEMORY_STORAGE_FRACTION,
    SPARK_CPU_CORES,
    AQE_ENABLED,
    AQE_COALESCE_PARTITIONS_ENABLED,
)


def create_spark() -> SparkSession:
    builder = (
        SparkSession.builder
        .appName(APP_NAME)
        .master(
            MASTER if MASTER != "local[*]"
            else f"local[{SPARK_CPU_CORES}]"
        )
        .config(
            "spark.sql.shuffle.partitions",
            str(SHUFFLE_PARTITIONS),
        )
        .config(
            "spark.sql.adaptive.enabled",
            str(AQE_ENABLED).lower(),
        )
        .config(
            "spark.sql.adaptive.coalescePartitions.enabled",
            str(AQE_COALESCE_PARTITIONS_ENABLED).lower(),
        )
        .config(
            "spark.driver.memory",
            SPARK_DRIVER_MEMORY,
        )
        .config(
            "spark.executor.memory",
            SPARK_EXECUTOR_MEMORY,
        )
        .config(
            "spark.memory.fraction",
            str(SPARK_MEMORY_FRACTION),
        )
        .config(
            "spark.memory.storageFraction",
            str(SPARK_MEMORY_STORAGE_FRACTION),
        )
    )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    return spark