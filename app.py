"""
Demo Streamlit — Amazon Hybrid Recommender System
Đồ án tốt nghiệp

Điểm quan trọng:
  - KHÔNG load model/catalog lúc mở app.
  - Chỉ load serving cache + chạy gợi ý sau khi người dùng đã có reviewerID.
  - Kết quả tự động cập nhật khi đổi reviewerID / Top-K / domain / alpha.

Chạy:
    streamlit run app.py
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.models.hybrid_recommender import HybridRecommender
from src.models.serving_store import ServingStore

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Amazon Hybrid Recommender — DATN",
    page_icon="🛍️",
    layout="wide",
)

st.title("🛍️ Amazon Hybrid Recommender System")
st.caption(
    "Đồ án tốt nghiệp: Mô hình gợi ý lai ghép — "
    "Collaborative Filtering (Popularity) + Content-Based (TF-IDF)"
)

# ──────────────────────────────────────────────────────────────────────────────
# Cached resources — chỉ chạy khi thực sự cần
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="⏳ Đang tải chỉ mục gợi ý...")
def load_model() -> HybridRecommender:
    rec = HybridRecommender()
    rec.load()
    return rec


@st.cache_data(show_spinner=False)
def run_recommendation(
    _rec: HybridRecommender,
    reviewer_id: str,
    top_k: int,
    domain,
    alpha: float,
    exclude_seen: bool,
):
    return _rec.recommend(
        user_raw_id=reviewer_id,
        top_k=top_k,
        domain=domain,
        alpha=alpha,
        exclude_seen=exclude_seen,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Cấu hình")

    top_k = st.slider("Top-K sản phẩm", 3, 20, 10)

    domain_opt = st.selectbox(
        "Lọc ngành hàng",
        ["Tất cả", "books", "electronics"],
    )
    domain = None if domain_opt == "Tất cả" else domain_opt

    alpha = st.slider(
        "Trọng số CF (α)",
        0.0, 1.0, 0.65, 0.05,
        help="Hybrid score = α·CF + (1−α)·CB",
    )
    st.caption(f"CF weight: **{alpha:.2f}** | CB weight: **{1-alpha:.2f}**")

    exclude_seen = st.checkbox("Loại sản phẩm đã tương tác", value=True)

    st.markdown("---")
    if st.button("🔄 Reload model", width="stretch"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# User selection
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("### 1. Người dùng")

sample_path = Path("data/demo_user_profile.json")
sample_users = []
if sample_path.exists():
    try:
        sample_users = json.loads(sample_path.read_text(encoding="utf-8"))["sample_users"]
    except Exception:
        pass

user_mode = st.radio(
    "Chọn cách nhập:",
    ["Tài khoản mẫu", "Nhập thủ công"],
    horizontal=True,
)

reviewer_id = ""
if user_mode == "Tài khoản mẫu" and sample_users:
    labels = {
        f"{u['reviewerID']} · {u['user_name']}": u for u in sample_users
    }
    chosen = st.selectbox("Chọn người dùng:", list(labels))
    info = labels[chosen]
    reviewer_id = info["reviewerID"]
    st.info(
        f"**Domain ưa thích:** {info.get('domain_preference', '—')}   "
        f"| {info.get('description', '')}"
    )
else:
    reviewer_id = st.text_input(
        "Nhập reviewerID:",
        value="",
        placeholder="ví dụ: A1IPEJDMQNXJYF",
        help="Nhấn Enter sau khi nhập — hệ thống sẽ tự động chạy gợi ý.",
    )

st.caption("💡 Kết quả tự động cập nhật ngay khi bạn nhập/chọn người dùng.")

# ──────────────────────────────────────────────────────────────────────────────
# Chưa có serving cache → hướng dẫn build một lần
# ──────────────────────────────────────────────────────────────────────────────

store = ServingStore()
if not store.is_built():
    st.warning(
        "Chưa có **serving cache** (chỉ mục gợi ý). "
        "Chỉ cần xây dựng một lần, sau đó app sẽ mở rất nhanh."
    )
    if st.button("⚡ Xây dựng chỉ mục ngay", type="primary"):
        progress = st.progress(0.0, text="Đang chuẩn bị...")

        def update(frac: float, message: str) -> None:
            progress.progress(min(1.0, max(0.0, frac)), text=message)

        try:
            info = store.build(progress=update, force=True)
        except Exception as exc:  # pragma: no cover - hiển thị lỗi cho người dùng
            st.error(f"Build thất bại: {exc}")
        else:
            st.cache_resource.clear()
            st.cache_data.clear()
            st.success(
                f"Đã tạo chỉ mục cho {info.get('catalog_size', 0):,} sản phẩm "
                f"trong {info.get('elapsed_s', 0)}s."
            )
            st.rerun()
    st.stop()

if not reviewer_id.strip():
    st.info("👆 Nhập hoặc chọn reviewerID để hệ thống bắt đầu gợi ý.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Load model (chỉ sau khi đã có user) + trạng thái
# ──────────────────────────────────────────────────────────────────────────────

try:
    rec = load_model()
except Exception as exc:  # pragma: no cover - hiển thị lỗi cho người dùng
    st.error(f"Không tải được mô hình: {exc}")
    st.stop()

status = rec.get_status()
mode = status["mode"]
if mode == "hybrid_pop_cb":
    st.success(
        f"**{status['mode_label']}** · "
        f"{status['catalog_size']:,} sản phẩm · "
        f"{status.get('num_users', 0):,} users · "
        f"TF-IDF vocab={status['vocab_size']:,} · "
        f"load={status['load_seconds']}s"
    )
elif mode in ("cf_only", "cb_only"):
    st.warning(status["mode_label"])
else:
    st.error(status["mode_label"])
    st.stop()

if not rec.user_known(reviewer_id.strip()):
    st.warning(
        f"Không tìm thấy `{reviewer_id.strip()}` trong mapping. "
        "Hệ thống vẫn gợi ý theo độ phổ biến của sản phẩm."
    )

# ──────────────────────────────────────────────────────────────────────────────
# Recommend — tự động chạy, không cần bấm nút
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("### 2. Kết quả gợi ý")

with st.spinner("Đang tính điểm Hybrid..."):
    results = run_recommendation(
        rec,
        reviewer_id.strip(),
        top_k,
        domain,
        alpha,
        exclude_seen,
    )

if not results:
    st.warning(
        "Không có kết quả. User có thể không có trong mapping "
        "hoặc không có sản phẩm phù hợp với bộ lọc hiện tại."
    )
    st.stop()

# ── Metrics row ──────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("Kết quả", len(results))
m2.metric("Best Hybrid Score", f"{results[0]['hybrid_score']:.4f}")
m3.metric("α (CF weight)", f"{alpha:.2f}")
m4.metric("Mode", mode.upper())

# ── Table ────────────────────────────────────────────────────────────────────
df = pd.DataFrame(results)
st.dataframe(
    df[["rank", "asin", "title", "brand", "category",
        "price", "hybrid_score", "cf_score", "cb_score"]],
    width="stretch",
    hide_index=True,
)

# ── Score breakdown chart ────────────────────────────────────────────────────
st.markdown("#### 📊 So sánh điểm CF vs CB vs Hybrid")
chart_df = df[["asin", "hybrid_score", "cf_score", "cb_score"]].set_index("asin")
st.bar_chart(chart_df)

# ── Explainability cards ─────────────────────────────────────────────────────
st.markdown("#### 💡 Chi tiết & Giải thích")
for item in results:
    with st.expander(
        f"#{item['rank']} — {item['title'][:60]} "
        f"| Hybrid: {item['hybrid_score']:.4f}",
        expanded=(item["rank"] <= 3),
    ):
        c1, c2 = st.columns([2, 1])
        c1.markdown(
            f"**ASIN:** `{item['asin']}`  \n"
            f"**Brand:** {item['brand']}  \n"
            f"**Category:** {item['category']}  \n"
            f"**Price:** {item['price']}"
        )
        c2.metric("🌟 Hybrid", item["hybrid_score"])
        c2.metric("👥 CF score", item["cf_score"])
        c2.metric("📝 CB score", item["cb_score"])
        st.caption(f"💬 {item['explanation']}")

# ── Download ─────────────────────────────────────────────────────────────────
st.download_button(
    "📥 Tải CSV",
    data=df.to_csv(index=False).encode("utf-8-sig"),
    file_name=f"hybrid_recs_{reviewer_id.strip()}.csv",
    mime="text/csv",
)

st.divider()
st.caption("Amazon Hybrid Recommender · DATN 2026 · Popularity + TF-IDF Fusion")
