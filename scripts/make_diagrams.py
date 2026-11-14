#!/usr/bin/env python
"""Sinh sơ đồ kiến trúc cho báo cáo (matplotlib, không cần graphviz).

Chạy:  .venv/bin/python scripts/make_diagrams.py
Xuất:  docs/figures/fig_6_1_pipeline.png ... fig_6_5_chon_mo_hinh.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#1a365d"
NAVY_D = "#0f2440"
NAVY_L = "#2a4a7f"
SLATE = "#475569"
SLATE_L = "#64748b"
GOLD = "#d69e2e"
GOLD_D = "#b7791f"
GOLD_L = "#ecc94b"
GREY = "#6b7280"
GREY_L = "#9ca3af"
LIGHT = "#f8fafc"
BLUE = "#dbeafe"
BLUE_D = "#93c5fd"
GREEN = "#dcfce7"
GREEN_D = "#86efac"
ROSE = "#fee2e2"
ROSE_D = "#fca5a5"
PURPLE = "#ede9fe"
PURPLE_D = "#c4b5fd"
WHITE = "#ffffff"
ARROW = "#475569"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.linewidth": 0,
    "figure.facecolor": "#ffffff",
})


def new_fig(w, h):
    # Canvas được thu nhỏ để chữ trong hình có kích thước thật ~7pt khi in ở khổ 16,2 cm
    fig, ax = plt.subplots(figsize=(w * 0.62, h * 0.62))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    return fig, ax


_PAIRS = []


def box(ax, x, y, w, h, text, fc=LIGHT, ec=NAVY, fs=8.6, bold=False,
        lw=1.2, color="#1f2937", align="center", pad=0.35, shadow=True, accent=None):
    if shadow:
        sp = FancyBboxPatch((x + 0.4, y - 0.4), w, h,
                            boxstyle=f"round,pad={pad},rounding_size=1.1",
                            linewidth=0, edgecolor="none", facecolor="#00000010", zorder=1)
        ax.add_patch(sp)
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad={pad},rounding_size=1.1",
                       linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(p)
    if accent is not None:
        bar = FancyBboxPatch((x, y + h - 3.0), w, 3.0,
                             boxstyle="round,pad=0.1,rounding_size=0.8",
                             linewidth=0, edgecolor="none", facecolor=accent, zorder=3)
        ax.add_patch(bar)
    t = ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=color, zorder=4,
                fontweight="bold" if bold else "normal",
                linespacing=1.42)
    _PAIRS.append((p, t))
    return p


def save(fig, name):
    """QA: cảnh báo nếu chữ tràn khung hoặc chữ đè chữ, rồi mới lưu."""
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    warn = []
    for p, t in _PAIRS:
        if not t.get_text().strip():
            continue
        tb = t.get_window_extent(ren)
        pb = p.get_window_extent(ren)
        ix = max(0.0, min(tb.x1, pb.x1) - max(tb.x0, pb.x0))
        iy = max(0.0, min(tb.y1, pb.y1) - max(tb.y0, pb.y0))
        inside = (ix * iy) / max(tb.width * tb.height, 1e-9)
        if inside < 0.985:
            warn.append(f"  TRAN KHUNG {inside*100:5.1f}%: {t.get_text()[:48]!r}")
    # Margin check disabled: title bar text intentionally sits at figure edge.
    # Inside-box check above already catches real text overflow.
    texts = [t for _, t in _PAIRS if t.get_text().strip()]
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            a = texts[i].get_window_extent(ren)
            b = texts[j].get_window_extent(ren)
            ix = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
            iy = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
            ov = (ix * iy) / max(min(a.width * a.height, b.width * b.height), 1e-9)
            if ov > 0.03:
                warn.append(f"  DE CHU {ov*100:5.1f}%: {texts[i].get_text()[:32]!r} <> {texts[j].get_text()[:32]!r}")
    if warn:
        print(f"[{name}] " + str(len(warn)) + " canh bao:")
        print("\n".join(warn[:12]))
    else:
        print(f"[{name}] OK")
    fig.savefig(OUT / name, dpi=220, facecolor="#ffffff", edgecolor="none")
    _PAIRS.clear()


def arrow(ax, p1, p2, style="-|>", color=ARROW, lw=1.3, rad=0.0, ls="-", dashed=False):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=13,
                        linewidth=lw, color=color, zorder=3,
                        connectionstyle=f"arc3,rad={rad}", linestyle="dashed" if dashed else ls,
                        shrinkA=2.0, shrinkB=2.0)
    ax.add_patch(a)


def title(ax, text, fs=13.5, sub=None):
    bar = FancyBboxPatch((0, 95.5), 100, 4.5,
                         boxstyle="square,pad=0", linewidth=0, edgecolor="none",
                         facecolor=NAVY, zorder=1)
    ax.add_patch(bar)
    ax.text(50, 97.75, text, ha="center", va="center", fontsize=fs,
            fontweight="bold", color="#ffffff", zorder=2)
    if sub:
        ax.text(50, 93.5, sub, ha="center", va="center", fontsize=8.6,
                color=SLATE, style="italic", zorder=2)


# ---------------------------------------------------------------- Hình 6.1
def fig_pipeline():
    fig, ax = new_fig(14, 7.2)
    title(ax, "Pipeline dữ liệu và luồng tín hiệu của hệ thống",
          sub="Hai nguồn dữ liệu → tiền xử lý → 4 nhánh tín hiệu → retrieval → fusion/rerank → đo & kiểm định")

    box(ax, 2, 82, 28, 12,
        "Amazon reviews\n(nguyteo/lastone)\n15,7M dòng tương tác", BLUE, NAVY)
    box(ax, 33, 82, 28, 12,
        "Metadata sản phẩm\n(nguyteo/finaldata)\n3,69M item sau rebuild", BLUE, NAVY)
    box(ax, 64, 82, 34, 12,
        "Tiền xử lý (implicit)\n• ≥ 3 tương tác, rating ≥ 4\n• lọc item/user rỗng",
        LIGHT, GOLD, fs=7.8)
    arrow(ax, (30, 88), (33, 88))
    arrow(ax, (61, 88), (64, 88))

    box(ax, 2, 62, 22, 12, "Chia theo user\nTrain  ·  Validation\nTest (chạy 1 lần)", LIGHT, NAVY)
    arrow(ax, (81, 82), (13, 74), rad=0.0)

    branches = [
        (2, "ALS\nHu et al. 2008\n32-d", BLUE),
        (26, "BPR\nRendle 2009\n32-d", BLUE),
        (50, "LightGCN\nHe et al. 2020\ngraph 2–3 lớp", BLUE),
        (74, "SBERT\nReimers & Gurevych\n768-d", GREEN),
    ]
    for x, t, fc in branches:
        box(ax, x, 42, 22, 13, t, fc, NAVY, fs=8.0)
        arrow(ax, (13, 62), (x + 11, 55))

    ax.text(50, 38.2, "Bốn nhánh tín hiệu độc lập — CF (hành vi) · graph (lan truyền) · content (văn bản)",
            ha="center", fontsize=8.4, color=GREY, style="italic")

    box(ax, 2, 22, 30, 12,
        "Sinh ứng viên (Stage 1)\nTop-K mỗi nhánh → union → dedup\nluôn chèn positive khi train", LIGHT, GOLD)
    box(ax, 35, 22, 30, 12,
        "Fusion / Rerank\n• Late Fusion V3\n• Early Fusion V3\n• RGH v4 (neo CF + gate)",
        LIGHT, GOLD, fs=8.0)
    box(ax, 68, 22, 30, 12,
        "Xếp hạng & đo\nHR@5/10/20 · NDCG@10 · MRR\nRecall@5/10/20", GREEN, NAVY)
    arrow(ax, (32, 28), (35, 28))
    arrow(ax, (65, 28), (68, 28))

    box(ax, 12, 3, 76, 13,
        "Kiểm định: bootstrap CI95 2.000 lần theo user  ·  Wilcoxon signed-rank (NDCG@10 per-user)\n"
        "·  McNemar exact (hit@10)  ·  tune trên validation, test chỉ chạy một lần",
        ROSE, "#9B2C2C", fs=8.4)
    arrow(ax, (50, 22), (50, 16))
    save(fig, "fig_6_1_pipeline.png")
    plt.close(fig)


# ---------------------------------------------------------------- Hình 6.2
def fig_rgh():
    fig, ax = new_fig(14, 8.8)
    title(ax, "Kiến trúc RGH v4: neo CF + residual gate (hai tầng)",
          sub="CF-Anchored Residual Gated Hybrid — kiến trúc nghiên cứu (HR@10 = 0,7983 vs LightGCN 0,7415)")

    ax.plot([1, 99], [56, 56], ls="--", lw=0.9, color="#C9C2B2")
    ax.text(2, 57.2, "STAGE 1 — RETRIEVAL", fontsize=8.4, color=GOLD, fontweight="bold")
    ax.text(2, 50.4, "STAGE 2 — RESIDUAL RERANKER", fontsize=8.4, color=GOLD, fontweight="bold")

    src = [
        (2, 84, "ALS\ntop-60", BLUE),
        (2, 72, "LightGCN\ntop-60", BLUE),
        (2, 60, "SBERT\ntop-60", GREEN),
    ]
    for x, y, t, fc in src:
        box(ax, x, y, 18, 9, t, fc, NAVY, fs=8.2)
    box(ax, 23, 60, 21, 33,
        "UNION ứng viên\n\n~140–220 item\n(1 positive + 99 negative\nkhi huấn luyện / đánh giá)",
        LIGHT, GOLD, fs=8.2)
    for _, y, _, _ in src:
        arrow(ax, (20, y + 4.5), (23, y + 4.5))

    box(ax, 48, 60, 22, 33,
        "Chuẩn hoá per-user\n\nz_cf  = z(s_ALS)\nz_bpr = z(s_BPR)\nz_gcn = z(s_GCN)\nz_txt = z(s_SBERT)",
        BLUE, NAVY, fs=8.2)
    arrow(ax, (44, 76.5), (48, 76.5))

    box(ax, 74, 60, 24, 33,
        "Vector đặc trưng 6-d\n\n[z_als, z_bpr, z_gcn, z_txt,\n z_als − z_gcn,  z_als · z_txt]\n\n+ trọng số học được",
        ROSE, "#9B2C2C", fs=8.2)
    arrow(ax, (70, 76.5), (74, 76.5))

    # Stage 2
    box(ax, 2, 30, 22, 17,
        "Linear anchor\n(neo CF)\n\na_cf·z_cf + a_gcn·z_gcn\n+ a_txt·z_txt\n0,735 / 0,499 / 0,299",
        LIGHT, GOLD, fs=8.0)
    box(ax, 2, 10, 22, 16,
        "Đặc trưng 6-d\n(features ở trên)\ndùng chung cho\ncross / deep / gate",
        LIGHT, NAVY, fs=8.0)
    arrow(ax, (86, 60), (86, 54.4), rad=0.0)
    arrow(ax, (86, 54.4), (13, 54.4), rad=0.0, style="-", lw=1.0)
    arrow(ax, (13, 54.4), (13, 47.5), rad=0.0)

    box(ax, 28, 36, 24, 11,
        "DCN cross × 2 lớp\nxₗ₊₁ = x₀ ⊙ tanh(Wₗxₗ) + xₗ\n(6 → 6, residual)", BLUE, NAVY, fs=8.0)
    box(ax, 28, 20, 24, 11,
        "Deep MLP\n6 → 64 → 32 → 1\nReLU · dropout 0,1", BLUE, NAVY, fs=8.0)
    box(ax, 28, 4, 24, 11,
        "Gate per-(user, item)\nσ(6 → 32 → 1)\nbias = −2,5 ⇒ g ≈ 0,08", ROSE, "#9B2C2C", fs=8.0)
    for y in (41.5, 25.5, 9.5):
        arrow(ax, (24, 18), (28, y), rad=-0.15)

    box(ax, 56, 20, 18, 11, "Δ = cross + deep\n(residual phi tuyến)", LIGHT, GOLD, fs=8.0)
    arrow(ax, (52, 41.5), (56, 29), rad=-0.2)
    arrow(ax, (52, 25.5), (56, 25.5))

    box(ax, 56, 4, 18, 11, "g · Δ\n(đóng góp có kiểm soát)", LIGHT, GOLD, fs=8.0)
    arrow(ax, (65, 20), (65, 15), rad=0.0)
    arrow(ax, (52, 9.5), (56, 9.5))

    box(ax, 78, 10, 20, 22,
        "score(u,i) =\nanchor + g·Δ\n\n→ xếp hạng\n100 ứng viên", GREEN, NAVY, fs=8.6, bold=True)
    arrow(ax, (74, 9.5), (78, 14), rad=0.1)
    # anchor đi ở làn riêng phía trên, tránh cắt qua cross network
    arrow(ax, (13, 47), (13, 48.2), rad=0.0)
    arrow(ax, (13, 48.2), (85, 48.2), rad=0.0, style="-", lw=1.0)
    arrow(ax, (85, 48.2), (85, 32.3), rad=0.0)

    ax.text(50, 1.4,
            "Khởi tạo a_cf = 0,60 / a_gcn = 0,30 / a_txt = 0,10 và gate ≈ 0  ⇒  lúc đầu mô hình gần như ALS;\n"
            "gate chỉ «mở» khi residual thực sự giảm loss.",
            ha="center", fontsize=8.0, color=GREY, style="italic")
    save(fig, "fig_6_2_rgh.png")
    plt.close(fig)


# ---------------------------------------------------------------- Hình 6.3
def fig_early_fusion():
    fig, ax = new_fig(14, 8.2)
    title(ax, "Early Fusion V3: DCN cross + deep MLP + listwise loss",
          sub="Mô hình triển khai chính — Recall@10 = 0,9004 (CI95 [0,8922; 0,9084])")

    box(ax, 2, 55, 26, 36,
        "Vector đầu vào 1.669-d\n\n"
        "u_als (32) ‖ i_als (32)\n"
        "u_bpr (32) ‖ i_bpr (32)\n"
        "u_txt (768) ‖ i_txt (768)\n"
        "z_cf, z_bpr, z_txt (3)\n"
        "log1p(pop), log1p(hist) (2)\n\n"
        "L2-norm cho embedding,\nz-score per-user cho điểm",
        BLUE, NAVY, fs=8.0)
    box(ax, 2, 34, 26, 17,
        "Vì sao 1.669 chiều?\n2·(32+32+768) + 5 = 1.669\n→ giữ nguyên thông tin thô\nthay vì chỉ 3 điểm fusion",
        LIGHT, GOLD, fs=7.9)

    box(ax, 32, 62, 30, 29,
        "Cross network — 3 lớp\n\n"
        "x₀ = input (1.669)\n"
        "xₗ₊₁ = x₀ ⊙ tanh(Wₗ xₗ) + xₗ\n"
        "Wₗ: 1.669 → 1.669, không bias\n\n"
        "học tương tác bậc cao giữa\nCF và content",
        BLUE, NAVY, fs=8.0)
    box(ax, 32, 26, 30, 31,
        "Deep MLP\n\n"
        "1.669 → 512 → ReLU → dropout 0,2\n"
        "→ 256 → ReLU → dropout 0,1\n"
        "→ 128 → ReLU → 1\n\n"
        "AdamW · lr 1e-3 · wd 1e-4\ngrad-clip 1,0 · ≤ 10 epoch",
        BLUE, NAVY, fs=8.0)
    arrow(ax, (28, 73), (32, 76))
    arrow(ax, (28, 73), (32, 42), rad=0.05)

    box(ax, 66, 55, 32, 13,
        "score = deep(x) + mean(cross)\n(residual readout, không cộng thêm bias)",
        LIGHT, GOLD, fs=8.4)
    arrow(ax, (62, 76), (66, 61.5), rad=-0.1)
    arrow(ax, (62, 41), (66, 58), rad=0.1)

    box(ax, 66, 36, 32, 15,
        "Listwise sampled softmax (T = 0,2)\n\n"
        "L = −log  exp(s_pos/T) / Σ exp(s_i/T)\n"
        "nhóm 1 positive + 40 negative\n(đánh giá: 1 + 99)",
        ROSE, "#9B2C2C", fs=8.0)
    arrow(ax, (82, 55), (82, 51))

    box(ax, 66, 8, 32, 22,
        "Vì sao listwise thay vì BCE?\n\n"
        "Mục tiêu là thứ tự trong nhóm ứng viên,\n"
        "không phải xác suất từng item.\n"
        "BCE chỉ đảm bảo pos > neg pointwise;\n"
        "listwise tối ưu trực tiếp phân bố hạng\n"
        "→ khớp với protocol 1 + 99.",
        GREEN, NAVY, fs=7.9)
    arrow(ax, (82, 36), (82, 30))

    ax.text(50, 1.6,
            "Đây là mô hình triển khai chính: Recall@10 = 0,9004 (CI95 [0,8922, 0,9084]) trên 5.000 user.",
            ha="center", fontsize=8.6, color=NAVY, fontweight="bold")
    save(fig, "fig_6_4_early_fusion.png")
    plt.close(fig)


# ---------------------------------------------------------------- Hình 6.4
def fig_late_fusion():
    fig, ax = new_fig(14, 6.4)
    title(ax, "Late Fusion V3: chuẩn hoá per-user rồi trộn tuyến tính",
          sub="Mô hình fusion tuyến tính — Recall@10 = 0,7736 (CI95 [0,7618; 0,7850])")

    srcs = [
        (2, "ALS\nnhân tử 32-d", BLUE),
        (2, "BPR\nnhân tử 32-d", BLUE),
        (2, "SBERT\n768-d", GREEN),
    ]
    ys = [70, 45, 20]
    for (x, t, fc), y in zip(srcs, ys):
        box(ax, x, y, 20, 16, t, fc, NAVY, fs=8.4)
        box(ax, 27, y, 18, 16, "z-score theo user\ntrên 100 ứng viên", LIGHT, GOLD, fs=8.2)
        arrow(ax, (22, y + 8), (27, y + 8))
        box(ax, 50, y, 14, 16, "", LIGHT, "#C9C2B2")
        arrow(ax, (45, y + 8), (50, y + 8))

    ax.text(57, 78, "trọng số\n(học trên validation)", ha="center", fontsize=8.0, color=GREY)
    for y, wt in zip(ys, ["w = 0,35", "w = 0,35", "w = 0,30"]):
        ax.text(57, y + 8, wt, ha="center", va="center", fontsize=8.6,
                color=NAVY, fontweight="bold", zorder=4)

    box(ax, 68, 42, 30, 16, "score = 0,35·z_ALS\n+ 0,35·z_BPR + 0,30·z_SBERT",
        LIGHT, GOLD, fs=9.0, bold=True)
    for y in ys:
        arrow(ax, (64, y + 8), (68, 50), rad=0.15)

    box(ax, 68, 18, 30, 16,
        "Xếp hạng 100 ứng viên\nRecall@10 = 0,7736\n(thấp hơn EF V3 ⇒ thiếu\nphi tuyến, đúng kỳ vọng)",
        GREEN, NAVY, fs=8.2)
    arrow(ax, (83, 42), (83, 34))

    ax.text(50, 5,
            "Ưu điểm: chỉ 3 tham số, không cần GPU, suy luận tức thời. Nhược điểm: không học được «khi nào tin CF hơn content».",
            ha="center", fontsize=8.2, color=GREY, style="italic")
    save(fig, "fig_6_3_late_fusion.png")
    plt.close(fig)


# ---------------------------------------------------------------- Hình 6.5
def fig_protocol():
    fig, ax = new_fig(14, 6.6)
    title(ax, "Protocol đánh giá: 1 positive + 99 negative, test chạy một lần",
          sub="Tune trên validation, test chạy 1 lần; bootstrap CI95 + Wilcoxon + McNemar theo từng user")

    box(ax, 2, 70, 26, 18,
        "Chia theo user\nTrain 80% · Validation 10%\nTest 10% (giữ kín đến cuối)",
        LIGHT, GOLD, fs=8.4)
    box(ax, 32, 70, 30, 18,
        "Mỗi user test u\n1 positive (item thật trong test)\n+ 99 negative lấy ngẫu nhiên\n(seed cố định, loại item đã tương tác)",
        BLUE, NAVY, fs=8.2)
    box(ax, 66, 70, 32, 18,
        "6 mô hình chấm cùng tập này\nALS · BPR · SBERT · ALS+BPR\nLate Fusion V3 · Early Fusion V3",
        LIGHT, NAVY, fs=8.2)
    arrow(ax, (28, 79), (32, 79))
    arrow(ax, (62, 79), (66, 79))

    box(ax, 2, 40, 26, 20,
        "Xếp hạng 100 ứng viên\nvị trí của positive = rank\n→ hit@K, NDCG@K, MRR",
        LIGHT, GOLD, fs=8.2)
    arrow(ax, (66, 70), (15, 60), rad=0.0)

    metrics = ["HR@5 / 10 / 20", "NDCG@5 / 10 / 20", "MRR", "Recall@5 / 10 / 20"]
    for k, m in enumerate(metrics):
        box(ax, 32 + (k % 2) * 17, 46 - (k // 2) * 11, 15, 8, m, GREEN, NAVY, fs=8.0)
    arrow(ax, (28, 50), (32, 50))

    box(ax, 68, 40, 30, 20,
        "Kiểm định\n\nbootstrap 2.000 lần theo user → CI95\nWilcoxon signed-rank (NDCG@10)\nMcNemar exact (hit@10)",
        ROSE, "#9B2C2C", fs=8.0)
    arrow(ax, (64, 50), (68, 50))

    box(ax, 20, 8, 60, 22,
        "Nguyên tắc chống leakage\n"
        "• SBERT user profile chỉ lấy item trong train history\n"
        "• Không dùng test để chọn hyperparameter / checkpoint\n"
        "• Cùng seed, cùng candidate cho mọi mô hình ⇒ so sánh ghép cặp hợp lệ\n"
        "• Recall@K ≡ HR@K vì mỗi user chỉ có 1 positive",
        LIGHT, GOLD, fs=8.2)
    arrow(ax, (50, 40), (50, 30))
    save(fig, "fig_7_1_protocol.png")
    plt.close(fig)


# ---------------------------------------------------------------- Hình 6.6
def fig_choice():
    fig, ax = new_fig(14, 6.6)
    title(ax, "Bản đồ quyết định: chọn mô hình nào cho mục tiêu nào",
          sub="Mỗi mục tiêu triển khai dẫn tới một mô hình phù hợp; EF V3 tối ưu đồng thời chất lượng và chi phí")

    box(ax, 30, 82, 40, 12,
        "Câu hỏi cần trả lời trước khi chọn mô hình\n(Chất lượng? Giải thích? Độ trễ? Nghiên cứu?)",
        LIGHT, GOLD, fs=8.4, bold=True)

    cols = [
        (2, "Chất lượng xếp hạng cao nhất\n→ Early Fusion V3",
         "Recall@10 = 0,9004\nCI95 [0,8922, 0,9084]\nDCN + MLP + listwise", GREEN, NAVY),
        (26, "Cần độ trễ thấp trên CPU\n→ LambdaRank (LightGBM)",
         "HR@10 = 0,8028\nretrain nhanh, < 5 ms/item\nnhưng không học end-to-end", LIGHT, NAVY),
        (50, "Cần kiến trúc có kiểm soát,\ngiải thích được → RGH v4",
         "HR@10 = 0,7983\nneo CF + gate + ablation\nphủ 100% user", ROSE, "#9B2C2C"),
        (74, "Chỉ cần baseline tin cậy\n→ ALS / BPR",
         "ALS 0,6752 · BPR 0,6340\nrẻ, ổn định, dễ tái lập\nnhưng coverage 62,29% user", BLUE, NAVY),
    ]
    for x, head, body, fc, ec in cols:
        box(ax, x, 55, 24, 15, head, fc, ec, fs=8.2, bold=True)
        box(ax, x, 26, 24, 25, body, "#FFFFFF", ec, fs=8.0)
        arrow(ax, (50, 82), (x + 12, 70), rad=0.08)

    ax.text(50, 16,
            "Quyết định của đồ án:  Early Fusion V3 = mô hình triển khai  ·  RGH v4 = kiến trúc nghiên cứu (không claim SOTA)\n"
            "Lý do: EF V3 thắng cả 5 baseline trên cùng protocol; RGH là nơi chứng minh từng thành phần\n"
            "(neo CF, gate, listwise) có đóng góp thật.",
            ha="center", fontsize=8.6, color=NAVY, fontweight="bold", linespacing=1.7)
    save(fig, "fig_6_5_chon_mo_hinh.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_pipeline()
    fig_rgh()
    fig_early_fusion()
    fig_late_fusion()
    fig_protocol()
    fig_choice()
    for p in sorted(OUT.glob("fig_6_*.png")):
        print(f"{p.name}: {p.stat().st_size/1e3:.0f} KB")
