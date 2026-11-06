# 1. Khởi tạo Deep Hybrid Recommender kết hợp mô hình ALS đã train với TF-IDF item features
deep_hybrid = DeepHybridRecommender(
    cf_model=als_model_instance,
    cb_item_features=item_features_csr,
    alpha=0.5,  # Cân bằng 50% CF latent factors và 50% CB features ở tầng Early Fusion
    beta=0.6    # Cân bằng 60% Pure CF score và 40% Early Fusion score ở tầng Late Fusion
)

# 2. Sinh Top-10 gợi ý cho user_id = 42
user_id = 42
top_items, top_scores = deep_hybrid.recommend(user_id=user_id, train_csr=user_item, n=10, filter_already_liked=True)

print(f"Top 10 Deep Hybrid Recommendations cho User {user_id}:")
for iid, score in zip(top_items, top_scores):
    print(f" - Item ID: {iid} | Deep Hybrid Score: {score:.4f}")