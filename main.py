"""
CLI Entrypoint for Amazon Hybrid Recommender System (DATN)
Sử dụng:
    python main.py --model hybrid --user A1IPEJDMQNXJYF --top_k 5
"""

import argparse
from pathlib import Path

from config import (
    MODEL_DIR,
    USER_FACTORS_PATH,
    ITEM_FACTORS_PATH,
    USER_MAPPING_PATH,
    ITEM_MAPPING_PATH,
)

from src.models.hybrid_recommender import HybridRecommender


def run_hybrid_cli(user_id: str, top_k: int, domain: str, alpha: float, exclude_seen: bool):
    print("=" * 65)
    print("🌟 DATN - AMAZON HYBRID RECOMMENDER (CLI DEMO)")
    print("=" * 65)

    recommender = HybridRecommender()
    try:
        recommender.load()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return

    status = recommender.get_status()

    print(f"Model status    : {status['mode_label']}")
    print(f"Target User ID  : {user_id}")
    print(f"Domain Filter   : {domain if domain else 'All'}")
    print(f"Top-K           : {top_k}")
    print(f"Hybrid Alpha    : {alpha} (CF) / {round(1.0 - alpha, 2)} (CB)")
    print("-" * 65)

    results = recommender.recommend(
        user_raw_id=user_id,
        top_k=top_k,
        domain=None if domain.lower() in ["all", "tất cả"] else domain,
        alpha=alpha,
        exclude_seen=exclude_seen,
    )

    if not results:
        print("Không tìm thấy kết quả đề xuất.")
        return

    print(f"\n{'Rank':<5} {'ASIN':<12} {'Hybrid':<8} {'CF':<8} {'CB':<8} {'Product Title'}")
    print("-" * 65)
    for item in results:
        title = (item['title'][:32] + '...') if len(item['title']) > 32 else item['title']
        print(f"#{item['rank']:<4} {item['asin']:<12} {item['hybrid_score']:<8.4f} {item['cf_score']:<8.4f} {item['cb_score']:<8.4f} {title}")

    print("\n" + "=" * 65)
    print("Để chạy giao diện Web Demo trực quan, hãy gõ lệnh:")
    print("   streamlit run app.py")
    print("=" * 65)


def run_cf_cli(user_id: str, top_k: int, domain: str, exclude_seen: bool):
    print("=" * 65)
    print("👥 DATN - COLLABORATIVE FILTERING (ALS)")
    print("=" * 65)

    required_paths = [
        USER_FACTORS_PATH,
        ITEM_FACTORS_PATH,
        USER_MAPPING_PATH,
        ITEM_MAPPING_PATH,
    ]
    missing = [str(p) for p in required_paths if not Path(p).exists()]
    if missing:
        print("ERROR: Chưa tìm thấy thư mục ALS output:")
        for p in missing:
            print(f"  - {p}")
        print("\nHãy đặt các file ALS output vào thư mục models/als hoặc sử dụng mô hình Hybrid:")
        print("   python main.py --model hybrid")
        return

    try:
        from src.spark_session import create_spark
        from src.recommender_cf import CollaborativeRecommender

        spark = create_spark()
        model = CollaborativeRecommender(spark=spark, model_dir=str(MODEL_DIR))
        recommendations = model.recommend(
            user_raw_id=user_id,
            N=top_k,
            domain=domain,
            exclude_seen=exclude_seen,
        )
        print(f"Gợi ý cho user {user_id}: {len(recommendations)} sản phẩm.")
    except Exception as e:
        print(f"Lỗi khi chạy Spark CF: {e}")


def main():
    parser = argparse.ArgumentParser(description="DATN Recommender System")
    parser.add_argument("--model", type=str, default="hybrid", choices=["hybrid", "cf"], help="Loại mô hình (hybrid hoặc cf)")
    parser.add_argument("--user", type=str, default="A1IPEJDMQNXJYF", help="Reviewer ID của người dùng")
    parser.add_argument("--top_k", type=int, default=5, help="Số sản phẩm gợi ý Top-K")
    parser.add_argument("--domain", type=str, default="All", help="Danh mục (All, Books, Electronics)")
    parser.add_argument("--alpha", type=float, default=0.65, help="Trọng số alpha cho Hybrid (0.0 đến 1.0)")
    parser.add_argument("--exclude_seen", action="store_true", default=True, help="Loại bỏ item đã tương tác")

    args = parser.parse_args()

    if args.model == "hybrid":
        run_hybrid_cli(
            user_id=args.user,
            top_k=args.top_k,
            domain=args.domain,
            alpha=args.alpha,
            exclude_seen=args.exclude_seen,
        )
    else:
        run_cf_cli(
            user_id=args.user,
            top_k=args.top_k,
            domain=args.domain,
            exclude_seen=args.exclude_seen,
        )


if __name__ == "__main__":
    main()
