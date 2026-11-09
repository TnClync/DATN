import React from 'react';

function Box({ title, sub, color, wide, mono }: { title: string; sub?: string; color: string; wide?: boolean; mono?: boolean }) {
  return (
    <div className={`rounded-lg border-2 ${color} ${wide ? 'px-5 py-2.5' : 'px-3 py-2'} text-center shadow-sm`}>
      <div className={`font-bold ${wide ? 'text-sm' : 'text-xs'}`}>{title}</div>
      {sub && <div className={`text-[9px] mt-0.5 opacity-75 ${mono ? 'font-mono' : ''}`}>{sub}</div>}
    </div>
  );
}

function Arrow({ vertical = false, label }: { vertical?: boolean; label?: string }) {
  if (vertical) {
    return (
      <div className="flex flex-col items-center py-1">
        {label && <div className="text-[9px] text-slate-400 mb-0.5 font-mono">{label}</div>}
        <div className="w-0.5 h-4 bg-slate-300" />
        <div className="w-0 h-0 border-l-4 border-r-4 border-t-[6px] border-l-transparent border-r-transparent border-t-slate-300" />
      </div>
    );
  }
  return (
    <div className="flex items-center px-1 self-center">
      {label && <div className="text-[9px] text-slate-400 mr-1 font-mono">{label}</div>}
      <div className="h-0.5 w-4 bg-slate-300" />
      <div className="w-0 h-0 border-t-4 border-b-4 border-l-[6px] border-t-transparent border-b-transparent border-l-slate-300" />
    </div>
  );
}

function Section({ label, n }: { label: string; n: string }) {
  return <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400 mb-2 mt-5">{n}. {label}</div>;
}

function Layer({ text, accent }: { text: string; accent?: string }) {
  return <div className={`text-[10px] font-mono px-2 py-1 rounded border ${accent || 'border-slate-200 bg-slate-50'} text-slate-700`}>{text}</div>;
}

function ModelCard({ name, type, color, children, note }: { name: string; type: string; color: string; children: React.ReactNode; note?: string }) {
  return (
    <div className={`rounded-xl border-2 ${color} p-3`}>
      <div className="flex items-center justify-between mb-2">
        <div className="font-bold text-xs">{name}</div>
        <span className="text-[9px] px-1.5 py-0.5 rounded bg-white/70 border border-slate-200 text-slate-500">{type}</span>
      </div>
      <div className="space-y-1.5">{children}</div>
      {note && <div className="mt-2 text-[9px] text-slate-500 leading-snug">{note}</div>}
    </div>
  );
}

export default function ModelArchitectureDiagram({ isDark }: { isDark: boolean }) {
  const card = isDark ? 'border-slate-700 bg-slate-900' : 'border-slate-200 bg-white';
  const sub = isDark ? 'text-slate-400' : 'text-slate-500';

  return (
    <div className={`rounded-xl border ${card} p-5`}>
      <h3 className="text-base font-bold mb-1">Model Architecture — Deep Hybrid Recommender (chi tiết)</h3>
      <p className={`text-xs ${sub} mb-3`}>Từ lịch sử user & catalog → embedding → fusion/ranking → Top-K. CF dùng <span className="font-mono">implicit</span>; deep hybrid là mô hình đề xuất.</p>

      <Section n="1" label="Đầu vào & không gian embedding" />
      <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
        <Box title="User History" sub="danh sách item_idx (đã tương tác)" color="border-blue-300 bg-blue-50" />
        <Box title="Candidate Items" sub="shared 200k catalog (eval) / union CF+SBERT+pop" color="border-blue-300 bg-blue-50" />
        <Box title="Ground Truth / Negatives" sub="train: 1 pos + 4 neg; eval: 1 pos + 99 neg" color="border-blue-300 bg-blue-50" />
      </div>
      <Arrow vertical label="lookup factors / encode" />
      <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
        <div className={`rounded-lg border p-2 ${isDark ? 'border-purple-700 bg-purple-950/30' : 'border-purple-200 bg-purple-50'}`}>
          <div className="text-xs font-bold text-purple-700 mb-1">CF / Graph Factors (float16 mmap)</div>
          <Layer text="ALS: user_f (u,32) · item_f (i,32)" accent="border-purple-200 bg-white" />
          <Layer text="BPR: user_f (u,32) · item_f (i,32)" accent="border-purple-200 bg-white" />
          <Layer text="LightGCN: graph_user (u,32), graph_item (i,32)" accent="border-purple-200 bg-white" />
          <div className={`text-[9px] mt-1 ${sub}`}>score_cf = dot(item_f[i], user_f[u]) (hoặc dot-product tương đương)</div>
        </div>
        <div className={`rounded-lg border p-2 ${isDark ? 'border-emerald-700 bg-emerald-950/30' : 'border-emerald-200 bg-emerald-50'}`}>
          <div className="text-xs font-bold text-emerald-700 mb-1">SBERT Content (all-MiniLM-L6-v2 → 768-d)</div>
          <Layer text="meta = SBERT(title+brand+category+description) → 384-d" accent="border-emerald-200 bg-white" />
          <Layer text="review = SBERT(review_content / reviewer text) → 384-d" accent="border-emerald-200 bg-white" />
          <Layer text="item_sbert = concat(meta, review) → 768-d (fp16)" accent="border-emerald-200 bg-white" />
          <div className={`text-[9px] mt-1 ${sub}`}>user_sbert = mean(item_sbert[history]) (L2-norm). item không review → copy meta. FAISS IndexFlatIP (cosine/IP) cho retrieval.</div>
        </div>
        <div className={`rounded-lg border p-2 ${isDark ? 'border-slate-600 bg-slate-800/40' : 'border-slate-200 bg-slate-50'}`}>
          <div className="text-xs font-bold mb-1">Điểm thành phần (scores)</div>
          <Layer text="CF score = item_cf[i] · user_cf[u]" accent="border-slate-200 bg-white" />
          <Layer text="SBERT score = item_sbert[i] · user_sbert[u]" accent="border-slate-200 bg-white" />
          <div className={`text-[9px] mt-1 ${sub}`}>Dùng làm feature/đầu vào cho các mô hình fusion bên dưới.</div>
        </div>
      </div>

      <Section n="2" label="Các mô hình fusion / deep hybrid (chi tiết)" />
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        <ModelCard name="Late Fusion (ALS + SBERT)" type="Hybrid (linear)" color="border-amber-300 bg-amber-50/60"
          note="α clamp ≤ 0.8 ở app. α=1.0 ⇒ fused = CF thuần (KHÔNG còn fusion) — trên full test best α=1.0 là degenerate ≡ ALS.">
          <Layer text="input: cf_score, sbert_score, α ∈ [0,1]" accent="border-amber-200 bg-white" />
          <div className="text-[10px] font-mono text-center py-1 rounded bg-white border border-amber-200">fused = α · CF_score + (1−α) · SBERT_score</div>
          <Layer text="rank candidates desc by fused → Top-K" accent="border-amber-200 bg-white" />
        </ModelCard>

        <ModelCard name="Light Gated Late Fusion (đề xuất deep hybrid)" type="Deep Hybrid" color="border-amber-400 bg-amber-50"
          note="Gate học trọng số fusion theo từng (user,item). Train: 1 pos + 4 neg, BCE/BCEWithLogits, Adam; eval batched.">
          <Layer text="u_cf = ALS user_f[u] (32) · i_cf = ALS item_f[i] (32)" accent="border-amber-200 bg-white" />
          <Layer text="i_sbert = SBERT item emb (768) · cf_score, sbert_score (scalars)" accent="border-amber-200 bg-white" />
          <Layer text="feat = concat([u_cf, i_cf, i_sbert]) → 832-d" accent="border-amber-200 bg-white" />
          <div className={`rounded border p-1.5 ${isDark ? 'border-amber-700 bg-slate-900/40' : 'border-amber-200 bg-white'}`}>
            <div className="text-[9px] font-bold text-amber-700 mb-1">Gate MLP (per (u,i))</div>
            <Layer text="Linear(832→128) → ReLU → Dropout(0.1)" accent="border-slate-200 bg-slate-50" />
            <Layer text="Linear(128→64) → ReLU → Dropout(0.1)" accent="border-slate-200 bg-slate-50" />
            <Layer text="Linear(64→1) → Sigmoid ⇒ α(u,i)" accent="border-slate-200 bg-slate-50" />
          </div>
          <div className="text-[10px] font-mono text-center py-1 rounded bg-white border border-amber-200">fused = α(u,i)·CF + (1−α(u,i))·SBERT</div>
        </ModelCard>

        <ModelCard name="Deep Early Fusion (DCN)" type="Deep Hybrid" color="border-amber-300 bg-amber-50/60"
          note="Early fusion: concat feature rồi học tương tác chéo (cross) + phi tuyến (deep). Head output là điểm xếp hạng.">
          <Layer text="x = concat([user_cf(32), item_cf(32), item_sbert(768)]) → 832-d" accent="border-amber-200 bg-white" />
          <div className="grid grid-cols-2 gap-1.5">
            <div className={`rounded border p-1.5 ${isDark ? 'border-amber-700 bg-slate-900/40' : 'border-amber-200 bg-white'}`}>
              <div className="text-[9px] font-bold mb-1">Cross Network (×4 layers)</div>
              <Layer text="x0 = x; for l in 1..4:" accent="border-slate-200 bg-slate-50" />
              <Layer text="  x_l = x0 ⊙ σ(W_l x_{l-1}) + x_{l-1}" accent="border-slate-200 bg-slate-50" />
              <Layer text="cross_out = mean(x_4, dim=-1)" accent="border-slate-200 bg-slate-50" />
            </div>
            <div className={`rounded border p-1.5 ${isDark ? 'border-amber-700 bg-slate-900/40' : 'border-amber-200 bg-white'}`}>
              <div className="text-[9px] font-bold mb-1">Deep MLP</div>
              <Layer text="Linear(832→256) → ReLU" accent="border-slate-200 bg-slate-50" />
              <Layer text="Linear(256→256) → ReLU" accent="border-slate-200 bg-slate-50" />
              <Layer text="Linear(256→128) → ReLU" accent="border-slate-200 bg-slate-50" />
              <Layer text="Linear(128→1) ⇒ deep_out" accent="border-slate-200 bg-slate-50" />
            </div>
          </div>
          <div className="text-[10px] font-mono text-center py-1 rounded bg-white border border-amber-200">score = cross_out + deep_out</div>
        </ModelCard>

        <ModelCard name="Two-Tower (Retrieval/Ranking)" type="Deep Hybrid" color="border-amber-300 bg-amber-50/60"
          note="Dùng LightGCN graph factors cho user/item CF-side. Item tower kết hợp graph + SBERT; similarity = dot product.">
          <div className="grid grid-cols-2 gap-1.5">
            <div className={`rounded border p-1.5 ${isDark ? 'border-blue-700 bg-slate-900/40' : 'border-blue-100 bg-white'}`}>
              <div className="text-[9px] font-bold text-blue-700 mb-1">User Tower</div>
              <Layer text="in: graph_user[u] (32)" accent="border-slate-200 bg-slate-50" />
              <Layer text="Linear(32→128) → ReLU → Dropout(0.1)" accent="border-slate-200 bg-slate-50" />
              <Layer text="Linear(128→64) ⇒ u_emb (64)" accent="border-slate-200 bg-slate-50" />
            </div>
            <div className={`rounded border p-1.5 ${isDark ? 'border-emerald-700 bg-slate-900/40' : 'border-emerald-100 bg-white'}`}>
              <div className="text-[9px] font-bold text-emerald-700 mb-1">Item Tower</div>
              <Layer text="in: concat(graph_item[i](32), i_sbert(768)) → 800" accent="border-slate-200 bg-slate-50" />
              <Layer text="Linear(800→256) → ReLU → Dropout(0.1)" accent="border-slate-200 bg-slate-50" />
              <Layer text="Linear(256→256) → ReLU → Dropout(0.1)" accent="border-slate-200 bg-slate-50" />
              <Layer text="Linear(256→64) ⇒ i_emb (64)" accent="border-slate-200 bg-slate-50" />
            </div>
          </div>
          <div className="text-[10px] font-mono text-center py-1 rounded bg-white border border-amber-200">score(u,i) = u_emb · i_emb</div>
        </ModelCard>

        <ModelCard name="Transformer Fusion (candidates)" type="Deep Hybrid" color="border-amber-300 bg-amber-50/60"
          note="Self-attention trên tập candidate của 1 user để mô hình hóa tương tác giữa các ứng viên; batch_first.">
          <Layer text="per-candidate feat = concat(graph_user[u](32), graph_item[c](32), sbert[c](768)) → 832" accent="border-amber-200 bg-white" />
          <Layer text="input sequence: (B, C, 832) for C candidates" accent="border-amber-200 bg-white" />
          <div className={`rounded border p-1.5 ${isDark ? 'border-amber-700 bg-slate-900/40' : 'border-amber-200 bg-white'}`}>
            <div className="text-[9px] font-bold mb-1">Transformer Encoder</div>
            <Layer text="proj: Linear(832→128) ⇒ d_model=128" accent="border-slate-200 bg-slate-50" />
            <Layer text="EncoderLayer ×2, nhead=4, d_ff=512, GELU, dropout=0.1, batch_first" accent="border-slate-200 bg-slate-50" />
            <Layer text="head: Linear(128→1) ⇒ score per candidate" accent="border-slate-200 bg-slate-50" />
          </div>
          <div className="text-[10px] font-mono text-center py-1 rounded bg-white border border-amber-200">scores = head(Encoder(proj(X))) → rank desc</div>
        </ModelCard>

        <ModelCard name="Ghi chú triển khai / serving" type="Info" color="border-slate-300 bg-slate-50">
          <Layer text="Serving load mmap float16; infer head input dim từ checkpoint (tương thích 384-d/768-d SBERT, 448-d/832-d head)." accent="border-slate-200 bg-white" />
          <Layer text="Pipeline: resolve user → CF (ALS/BPR) + SBERT (FAISS/profile) → union candidates → fusion head → Top-K + metadata (SQLite)." accent="border-slate-200 bg-white" />
          <Layer text="API: /recommend, /simulate, /metrics, /health, /models. Alpha clamp ≤ 0.8; α=1 pure CF." accent="border-slate-200 bg-white" />
          <div className={`text-[9px] ${sub}`}>Kích thước kernel training (tham chiếu): LightGate 832→128→64→1; DCN cross×4+deep 256/256/128; Two-Tower out 64; Transformer d_model 128, 2 layers, 4 heads. Graph Light Gated tương tự Light Gated nhưng dùng LightGCN factors.</div>
        </ModelCard>
      </div>

      <Section n="3" label="Đầu ra & đánh giá" />
      <div className="flex flex-col md:flex-row items-stretch gap-2">
        <Box title="Xếp hạng Top-K" sub="re-rank theo fused/model score (desc), trả k items" color="border-rose-300 bg-rose-50" wide />
        <Arrow label="evaluate" />
        <Box title="Metrics (full test)" sub="HR@5/10/20 · NDCG@5/10/20 · MRR · Recall@5/10/20" color="border-rose-300 bg-rose-50" wide />
      </div>
      <div className={`mt-4 pt-3 border-t ${isDark ? 'border-slate-700' : 'border-slate-200'} text-[10px] ${sub} space-y-1`}>
        <div><strong>Protocol:</strong> 1 positive + 99 negatives / user, shared 200k catalog, leak-free SBERT profile, <strong>299,455</strong> eligible users (min_interactions=3). Đánh giá toàn bộ test set.</div>
        <div><strong>Baseline (full test):</strong> ALS HR@10 <strong>0.6153</strong> / NDCG@10 0.4407 (best overall). <strong>Light Gated</strong> HR@10 <strong>0.5245</strong> / NDCG@10 0.3499 (best deep hybrid). Late Fusion best α=1.0 ≡ ALS (degenerate, không fusion).</div>
        <div><strong>Artifacts:</strong> CF <span className="font-mono">factors/*.npy(f16)</span>, SBERT <span className="font-mono">sbert_item_embeddings_f16.npy</span> (384-d local / 768-d Kaggle target), heads <span className="font-mono">models/*.pt</span>, FAISS index.</div>
      </div>
    </div>
  );
}
