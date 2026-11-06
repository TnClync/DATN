#!/usr/bin/env python3
"""Generate thesis report (Đồ án.docx)."""
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from pathlib import Path

doc = Document()

# ── Styles ──────────────────────────────────────────────────────────────────
style = doc.styles['Normal']
style.font.name = 'Times New Roman'
style.font.size = Pt(12)

def add_heading(text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.name = 'Times New Roman'
    return h

def add_para(text, bold=False, italic=False, size=12, align=None):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    run.font.size = Pt(size)
    if align:
        p.alignment = align
    return p

def add_table(headers, rows, widths=None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = h
        for p in hdr[i].paragraphs:
            for r in p.runs:
                r.bold = True
                r.font.size = Pt(10)
    for row in rows:
        cells = table.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = str(val)
            for p in cells[i].paragraphs:
                for r in p.runs:
                    r.font.size = Pt(10)
    return table

# ── Title ───────────────────────────────────────────────────────────────────
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title.add_run('ĐỒ ÁN TỐT NGHIỆP')
run.bold = True
run.font.size = Pt(18)
run.font.name = 'Times New Roman'

subtitle = doc.add_paragraph()
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = subtitle.add_run('Xây dựng mô hình lai ghép cho hệ thống khuyến nghị sản phẩm (Amazon)')
run.bold = True
run.font.size = Pt(14)

doc.add_paragraph()

# ── 1. Tổng quan ────────────────────────────────────────────────────────────
add_heading('1. Tổng quan', level=1)
add_para('Hệ thống gợi ý lai ghép (Hybrid Recommender System) kết hợp Collaborative Filtering (CF) '
         'và Content-Based Filtering (CB) trên dữ liệu Amazon. Train trên Kaggle 2×T4 GPU với datasets '
         'nguynteo/lastone (interactions) và nguynteo/lastreview (review content).')

# ── 2. Các model đã train ───────────────────────────────────────────────────
add_heading('2. Các model đã train thành công', level=1)

add_heading('2.1 Collaborative Filtering (CF)', level=2)
add_table(
    ['Model', 'File', 'Kích thước', 'RMSE', 'MAE', 'HR@10', 'NDCG@10', 'MRR'],
    [
        ['MF-bias', 'models/mf_bias/', '209MB', '0.3727', '0.2869', '0.6470', '0.4368', '0.3705'],
        ['ALS (PyTorch, f=32)', 'models/als_torch/', '~1MB', '3.8359', '3.8165', '0.9725', '0.7710', '0.7062'],
        ['ALS (PyTorch, f=64)', 'models/als_torch/', '~1MB', '3.8331', '3.8134', '1.0000', '0.9252', '0.8990'],
        ['ALS Best (f=64, iters=20)', 'models/als_best/', '24MB', '1282051.87', '2333.93', '0.1644', '0.0834', '0.0590'],
        ['ALS Last (f=128, iters=20)', 'N/A', '—', '4.7383', '4.7186', '0.0534', '0.0367', '0.0316'],
        ['BPR-MF', 'N/A (lost)', '—', '—', '—', '0.0184', '0.0104', '—'],
    ]
)

add_heading('2.2 Content-Based (CB)', level=2)
add_table(
    ['Model', 'File', 'Kích thước', 'HR@10', 'NDCG@10', 'MRR'],
    [
        ['SBERT Content', 'models/sbert/', '622MB', '0.0000', '0.0000', '0.0000'],
    ]
)
add_para('SBERT (all-MiniLM-L6-v2) trên review content, aggregated per item_idx. 424,361 items × 384 dim.', italic=True, size=10)

add_heading('2.3 Hybrid — Deep (có file weights)', level=2)
add_table(
    ['Model', 'File', 'Kích thước', 'RMSE', 'MAE', 'HR@10', 'NDCG@10', 'MRR'],
    [
        ['Gated Hybrid', 'models/gated_hybrid/', '1.6GB', '—', '—', '—', '—', '—'],
        ['DCN-v2', 'models/dcnv2/', '2.7GB', '0.4445', '0.2536', '0.2632', '0.1193', '0.1018'],
        ['TwoTower', 'models/twotower/', '2.7GB', '0.4385', '0.2491', '0.1974', '0.0936', '0.0902'],
        ['ResidualCascade', 'models/residual_cascade/', '2.8GB', '0.2647*', '—', '0.1632', '0.0863', '0.0893'],
        ['AutoInt', 'models/autoint/', '2.7GB', '0.4629', '0.2455', '0.2289', '0.1087', '0.1005'],
        ['Deep Fusion MLP', 'models/deep_fusion/', '1.1MB', '—', '—', '—', '—', '—'],
    ]
)
add_para('* ResidualCascade: train RMSE only', italic=True, size=9)

add_heading('2.4 Hybrid — Linear', level=2)
add_table(
    ['Model', 'HR@10', 'NDCG@10', 'MRR', 'Ghi chú'],
    [
        ['Popularity', '0.6560', '0.4680', '0.4094', 'Strong baseline'],
        ['LateFusion (α=0.5)', '0.6515', '0.4324', '0.3633', 'Blend CF+CB'],
        ['LateFusion (α=0.7)', '0.6515', '0.4321', '0.3631', 'Blend CF+CB'],
        ['EarlyFusion', '0.6470', '0.4346', '0.3678', 'Linear + interaction'],
    ]
)

add_heading('2.5 Hybrid — Deep (chưa train, chỉ có eval metrics)', level=2)
add_table(
    ['Model', 'HR@10', 'NDCG@10', 'MRR', 'Delta vs Baseline'],
    [
        ['NeuMF-Hybrid', '0.1974', '0.1022', '0.1008', '-2.6%'],
        ['Multi-Affinity Bilinear', '0.1868', '0.0876', '0.0829', '-5%'],
        ['Two-Tower Contrastive', '0.1974', '0.0936', '0.0902', '-2.6%'],
    ]
)

# ── 3. Kiến trúc model ─────────────────────────────────────────────────────
add_heading('3. Kiến trúc các model', level=1)

add_heading('3.1 MF-bias', level=2)
add_para('Matrix Factorization with biases: score = user_emb · item_emb + user_bias + item_bias + global_bias')
add_para('factors=64, epochs=5, lr=0.005, weight_decay=1e-5')

add_heading('3.2 SBERT Content', level=2)
add_para('Sentence-BERT (all-MiniLM-L6-v2) encode review content → 384-dim embeddings, aggregated per item_idx. User profile = mean of seen items\' embeddings.')

add_heading('3.3 DCN-v2 (Deep & Cross Network)', level=2)
add_para('Cross Network (3 layers): x_{l+1} = x0 ⊙ (W_l·x_l + b_l) + x_l')
add_para('Deep Network: 256→128→64 (ReLU, Dropout)')
add_para('Predict head: Linear(96, 1) — concat [cross_out(64) + deep_out(32)]')

add_heading('3.4 TwoTower', level=2)
add_para('User Tower: 32→128→64 | Item Tower: 32→128→64')
add_para('Score = scale × dot(User, Item)')

add_heading('3.5 ResidualCascade', level=2)
add_para('Coarse: 896→64→1 → Score_c')
add_para('Fine: 897→128→64→1 (input = features + coarse score)')
add_para('Output = Fine + Coarse (residual)')

add_heading('3.6 AutoInt (Self-Attention)', level=2)
add_para('Projection: 896→128 → Multi-Head Attention (4 heads, 2 layers) → Deep MLP (128→64)')
add_para('Score = Linear(192, 1) — concat [attention(128) + deep(64)]')

add_heading('3.7 Gated Hybrid', level=2)
add_para('Gate: 896→128→64 → Sigmoid | Transform: 896→64 → ReLU')
add_para('Gated = Gate ⊙ Transform → Linear(64,32) → ReLU → Linear(32,1)')

add_heading('3.8 LateFusion', level=2)
add_para('Score = α·norm(CF) + (1-α)·norm(CB), α ∈ {0.5, 0.7}')

add_heading('3.9 EarlyFusion', level=2)
add_para('Score = 0.4·CF_norm + 0.3·CB_norm + 0.3·CF_norm·CB_norm')

# ── 4. Khuyến nghị ──────────────────────────────────────────────────────────
add_heading('4. Khuyến nghị', level=1)
add_table(
    ['Vai trò', 'Model', 'Lý do'],
    [
        ['Baseline hybrid', 'EarlyFusion (HR@10=0.6470)', 'Hybrid baseline hợp lý'],
        ['Recommended deep', 'DCN-v2 (HR@10=0.2632, +33%)', 'Cross-features phi tuyến, tốt nhất'],
        ['Best linear hybrid', 'LateFusion (HR@10=0.6515)', 'Blend CF+CB, dễ giải thích'],
        ['Rating prediction', 'MF-bias (RMSE 0.3727)', 'Tốt nhất cho rating'],
        ['Content-only', 'SBERT + FAISS', 'Tìm kiếm nhanh trên 424K items'],
    ]
)

# ── 5. Web Inference ────────────────────────────────────────────────────────
add_heading('5. Web Inference (FastAPI)', level=1)
add_para('Chạy: uv run uvicorn api:app --host 0.0.0.0 --port 8000')
add_para('Web UI: http://localhost:8000 | Swagger: http://localhost:8000/docs')
add_table(
    ['Method', 'Path', 'Mô tả'],
    [
        ['GET', '/', 'Trang chủ (HTML)'],
        ['GET', '/health', 'Health check'],
        ['GET', '/models', 'List available models'],
        ['POST', '/recommend', 'Gợi ý (customer_id, model, top_k, alpha)'],
        ['GET', '/compare', 'Bảng so sánh 14 model'],
        ['GET', '/customer/{id}', 'Thông tin customer'],
    ]
)
add_para('Models available: gated_hybrid, dcnv2, twotower, residual_cascade, late_fusion')
add_para('Customer ID → user_idx mapping (SQLite, 22M users)')
add_para('Pre-load: lazy (load on first request)')
add_para('FAISS: tìm kiếm trên 424K SBERT items')

# ── 6. Cấu trúc thư mục ─────────────────────────────────────────────────────
add_heading('6. Cấu trúc thư mục', level=1)
add_para('DATN/')
add_para('├── api.py                          # FastAPI backend (primary web interface)')
add_para('├── src/models/                     # Modular model classes')
add_para('├── scripts/demo_inference.py       # CLI demo')
add_para('├── models/                         # Trained models')
add_para('├── docs/                           # Documentation + comparison')
add_para('└── data/                           # Demo user profiles')

# ── 7. Datasets ─────────────────────────────────────────────────────────────
add_heading('7. Datasets', level=1)
add_para('- nguynteo/lastone — interactions (user_idx, item_idx, rating, ...)')
add_para('- nguynteo/lastreview — review content (review_content per item_idx)')

# ── Save ────────────────────────────────────────────────────────────────────
out_path = Path('/home/clyde/DATN/docs/Đồ án.docx')
doc.save(str(out_path))
print(f'Saved to {out_path}')
