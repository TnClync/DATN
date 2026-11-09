import React from 'react';
import { Layers, Cpu, Database, Sparkles, GitMerge, BarChart3, Zap, Sigma, Binary } from 'lucide-react';

type Tone = 'blue' | 'purple' | 'emerald' | 'amber' | 'rose' | 'slate' | 'indigo';

function toneClasses(t: Tone, isDark: boolean) {
  const m: Record<Tone, { border: string; bg: string; head: string; chip: string; chipBorder: string }> = {
    blue:    { border: isDark ? 'border-blue-500/40' : 'border-blue-200',    bg: isDark ? 'bg-blue-500/10' : 'bg-gradient-to-br from-blue-50 to-white',       head: isDark ? 'text-blue-300' : 'text-blue-700',    chip: isDark ? 'bg-blue-950/40' : 'bg-blue-50/80',    chipBorder: isDark ? 'border-blue-500/30' : 'border-blue-200' },
    purple:  { border: isDark ? 'border-purple-500/40' : 'border-purple-200',bg: isDark ? 'bg-purple-500/10' : 'bg-gradient-to-br from-purple-50 to-white',   head: isDark ? 'text-purple-300' : 'text-purple-700',chip: isDark ? 'bg-purple-950/40' : 'bg-purple-50/80',chipBorder: isDark ? 'border-purple-500/30' : 'border-purple-200' },
    emerald: { border: isDark ? 'border-emerald-500/40' : 'border-emerald-200',bg:isDark ? 'bg-emerald-500/10' : 'bg-gradient-to-br from-emerald-50 to-white', head: isDark ? 'text-emerald-300' : 'text-emerald-700',chip:isDark ? 'bg-emerald-950/40' : 'bg-emerald-50/80',chipBorder:isDark ? 'border-emerald-500/30' : 'border-emerald-200' },
    amber:   { border: isDark ? 'border-amber-500/40' : 'border-amber-200',  bg: isDark ? 'bg-amber-500/10' : 'bg-gradient-to-br from-amber-50 to-white',     head: isDark ? 'text-amber-300' : 'text-amber-700',  chip: isDark ? 'bg-amber-950/30' : 'bg-amber-50/70',  chipBorder: isDark ? 'border-amber-500/30' : 'border-amber-200' },
    rose:    { border: isDark ? 'border-rose-500/40' : 'border-rose-200',    bg: isDark ? 'bg-rose-500/10' : 'bg-gradient-to-br from-rose-50 to-white',       head: isDark ? 'text-rose-300' : 'text-rose-700',    chip: isDark ? 'bg-rose-950/40' : 'bg-rose-50/80',    chipBorder: isDark ? 'border-rose-500/30' : 'border-rose-200' },
    slate:   { border: isDark ? 'border-slate-600' : 'border-slate-200',     bg: isDark ? 'bg-slate-800/50' : 'bg-gradient-to-br from-slate-50 to-white',    head: isDark ? 'text-slate-200' : 'text-slate-700',  chip: isDark ? 'bg-slate-800' : 'bg-slate-50',        chipBorder: isDark ? 'border-slate-600' : 'border-slate-200' },
    indigo:  { border: isDark ? 'border-indigo-500/40' : 'border-indigo-200',bg: isDark ? 'bg-indigo-500/10' : 'bg-gradient-to-br from-indigo-50 to-white',   head: isDark ? 'text-indigo-300' : 'text-indigo-700',chip: isDark ? 'bg-indigo-950/40' : 'bg-indigo-50/80',chipBorder: isDark ? 'border-indigo-500/30' : 'border-indigo-200' },
  };
  return m[t];
}

function FlowArrow({ label, isDark }: { label?: string; isDark: boolean }) {
  return (
    <div className="flex flex-col items-center py-1.5">
      {label && <span className={`text-[9px] font-mono px-2 py-0.5 rounded-full mb-1 ${isDark ? 'bg-slate-800 text-slate-400 border border-slate-700' : 'bg-slate-100 text-slate-500 border border-slate-200'}`}>{label}</span>}
      <div className={`w-px h-4 ${isDark ? 'bg-slate-600' : 'bg-slate-300'}`} />
      <div className={`w-0 h-0 border-l-[5px] border-r-[5px] border-t-[7px] border-l-transparent border-r-transparent ${isDark ? 'border-t-slate-500' : 'border-t-slate-400'}`} />
    </div>
  );
}

function Section({ n, label, icon: Icon, isDark }: { n: string; label: string; icon: React.ElementType; isDark: boolean }) {
  return (
    <div className="flex items-center gap-2 mt-6 mb-3">
      <div className={`flex items-center justify-center h-7 w-7 rounded-lg ${isDark ? 'bg-indigo-500/20 text-indigo-300' : 'bg-indigo-100 text-indigo-700'}`}>
        <Icon className="h-4 w-4" />
      </div>
      <div className={`text-[11px] font-bold uppercase tracking-wider ${isDark ? 'text-slate-300' : 'text-slate-600'}`}>{n}. {label}</div>
      <div className={`flex-1 h-px ${isDark ? 'bg-slate-700' : 'bg-slate-200'}`} />
    </div>
  );
}

function Chip({ text, tone = 'slate', isDark }: { text: string; tone?: Tone; isDark: boolean }) {
  const c = toneClasses(tone, isDark);
  return (
    <div className={`text-[10px] font-mono px-2.5 py-1.5 rounded-md border-l-2 ${c.chipBorder} ${isDark ? 'bg-slate-800/70 text-slate-200 border border-slate-700/80' : 'bg-white/80 text-slate-700 border border-slate-200/80'} shadow-[0_1px_0_rgba(0,0,0,0.03)]`}>
      {text}
    </div>
  );
}

function Formula({ text, isDark, highlight }: { text: string; isDark: boolean; highlight?: boolean }) {
  return (
    <div className={`text-[10px] font-mono text-center px-2 py-1.5 rounded-md border ${highlight ? (isDark ? 'border-indigo-400/40 bg-indigo-500/15 text-indigo-200' : 'border-indigo-200 bg-indigo-50 text-indigo-800') : (isDark ? 'border-slate-600 bg-slate-800 text-slate-200' : 'border-slate-200 bg-slate-50 text-slate-700')}`}>
      {text}
    </div>
  );
}

function ModelCard({ name, type, tone, recommended, children, note, isDark }: { name: string; type: string; tone: Tone; recommended?: boolean; children: React.ReactNode; note?: string; isDark: boolean }) {
  const c = toneClasses(tone, isDark);
  return (
    <div className={`relative rounded-xl border ${c.border} ${c.bg} p-3.5 shadow-sm ${recommended ? (isDark ? 'ring-1 ring-indigo-400/40' : 'ring-1 ring-indigo-300/60') : ''}`}>
      {recommended && <div className="absolute -top-2 right-3 text-[9px] font-bold px-2 py-0.5 rounded-full bg-indigo-600 text-white shadow">ĐỀ XUẤT</div>}
      <div className="flex items-start justify-between gap-2 mb-2.5">
        <div className={`font-bold text-[13px] leading-snug ${isDark ? 'text-slate-100' : 'text-slate-800'}`}>{name}</div>
        <span className={`shrink-0 text-[9px] font-semibold px-2 py-0.5 rounded-full border ${isDark ? 'bg-slate-800 border-slate-600 text-slate-300' : 'bg-white/80 border-slate-200 text-slate-600'}`}>{type}</span>
      </div>
      <div className="space-y-1.5">{children}</div>
      {note && <div className={`mt-2.5 pt-2 border-t text-[10px] leading-snug ${isDark ? 'border-slate-700 text-slate-400' : 'border-slate-200/80 text-slate-500'}`}>{note}</div>}
    </div>
  );
}

function MiniBox({ title, sub, tone, isDark, wide }: { title: string; sub: string; tone: Tone; isDark: boolean; wide?: boolean }) {
  const c = toneClasses(tone, isDark);
  return (
    <div className={`rounded-xl border ${c.border} ${c.bg} ${wide ? 'px-5 py-3' : 'px-3 py-2.5'} text-center shadow-sm`}>
      <div className={`font-semibold text-xs ${isDark ? 'text-slate-100' : 'text-slate-800'}`}>{title}</div>
      <div className={`text-[10px] mt-0.5 ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>{sub}</div>
    </div>
  );
}

export default function ModelArchitectureDiagram({ isDark }: { isDark: boolean }) {
  const card = isDark ? 'border-slate-700/80 bg-slate-900' : 'border-slate-200 bg-white';
  const sub = isDark ? 'text-slate-400' : 'text-slate-500';
  const panel = (t: Tone, extra = '') => {
    const c = toneClasses(t, isDark);
    return `rounded-xl border ${c.border} ${c.bg} p-3 ${extra}`;
  };

  return (
    <div className={`rounded-2xl border ${card} p-5 shadow-sm`}>
      <div className={`rounded-xl p-4 mb-2 ${isDark ? 'bg-gradient-to-r from-indigo-600/20 via-violet-600/10 to-transparent border border-indigo-500/20' : 'bg-gradient-to-r from-indigo-50 via-violet-50/60 to-white border border-indigo-100'}`}>
        <div className="flex flex-wrap items-center gap-2 mb-1">
          <Sparkles className={`h-5 w-5 ${isDark ? 'text-indigo-300' : 'text-indigo-600'}`} />
          <h3 className="text-base font-bold">Model Architecture — Deep Hybrid Recommender</h3>
          <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-amber-500/15 text-amber-700 border border-amber-300/50">chi tiết</span>
        </div>
        <p className={`text-xs ${sub}`}>Pipeline: <span className="font-mono">History/Catalog → Embeddings → Fusion/Ranking → Top-K</span>. CF dùng <span className="font-mono">implicit</span>; <strong>deep hybrid</strong> là mô hình đề xuất.</p>
        <div className="flex flex-wrap gap-1.5 mt-2">
          {['CF 32-d', 'LightGCN 32-d', 'SBERT 384+384→768', 'α·CF+(1−α)·SBERT', 'Gate/DCN/2-Tower/Transformer'].map(t => (
            <span key={t} className={`text-[9px] font-mono px-2 py-0.5 rounded-full border ${isDark ? 'bg-slate-800/80 border-slate-600 text-slate-300' : 'bg-white/80 border-slate-200 text-slate-600'}`}>{t}</span>
          ))}
        </div>
      </div>

      <Section n="1" label="Đầu vào & không gian embedding" icon={Database} isDark={isDark} />
      <div className="grid grid-cols-1 md:grid-cols-3 gap-2.5">
        <MiniBox title="User History" sub="danh sách item_idx (đã tương tác)" tone="blue" isDark={isDark} />
        <MiniBox title="Candidate Items" sub="shared 200k (eval) / union CF+SBERT+pop" tone="blue" isDark={isDark} />
        <MiniBox title="GT / Negatives" sub="train: 1 pos + 4 neg · eval: 1 pos + 99 neg" tone="blue" isDark={isDark} />
      </div>
      <FlowArrow label="lookup factors / encode" isDark={isDark} />
      <div className="grid grid-cols-1 md:grid-cols-3 gap-2.5">
        <div className={panel('purple')}>
          <div className={`flex items-center gap-1.5 text-xs font-bold mb-2 ${isDark ? 'text-purple-300' : 'text-purple-700'}`}><Cpu className="h-3.5 w-3.5" />CF / Graph Factors <span className="font-normal opacity-70">(f16 mmap)</span></div>
          <div className="space-y-1.5">
            <Chip text="ALS: user_f (u,32) · item_f (i,32)" tone="purple" isDark={isDark} />
            <Chip text="BPR: user_f (u,32) · item_f (i,32)" tone="purple" isDark={isDark} />
            <Chip text="LightGCN: graph_user(u,32), graph_item(i,32)" tone="purple" isDark={isDark} />
          </div>
          <div className={`text-[10px] mt-2 ${sub}`}>score_cf = dot(item_f[i], user_f[u])</div>
        </div>
        <div className={panel('emerald')}>
          <div className={`flex items-center gap-1.5 text-xs font-bold mb-2 ${isDark ? 'text-emerald-300' : 'text-emerald-700'}`}><Binary className="h-3.5 w-3.5" />SBERT Content <span className="font-normal opacity-70">(768-d)</span></div>
          <div className="space-y-1.5">
            <Chip text="meta = SBERT(title+brand+cat+desc) → 384-d" tone="emerald" isDark={isDark} />
            <Chip text="review = SBERT(review_content) → 384-d" tone="emerald" isDark={isDark} />
            <Chip text="item_sbert = concat(meta,review) → 768-d (fp16)" tone="emerald" isDark={isDark} />
          </div>
          <div className={`text-[10px] mt-2 ${sub}`}>user_sbert = mean(item_sbert[history]) (L2-norm). No review → copy meta. FAISS IndexFlatIP (IP/cosine).</div>
        </div>
        <div className={panel('slate')}>
          <div className={`flex items-center gap-1.5 text-xs font-bold mb-2 ${isDark ? 'text-slate-200' : 'text-slate-700'}`}><Sigma className="h-3.5 w-3.5" />Điểm thành phần</div>
          <div className="space-y-1.5">
            <Chip text="CF score = item_cf[i] · user_cf[u]" tone="slate" isDark={isDark} />
            <Chip text="SBERT score = item_sbert[i] · user_sbert[u]" tone="slate" isDark={isDark} />
          </div>
          <div className={`text-[10px] mt-2 ${sub}`}>Feature/đầu vào cho fusion models bên dưới.</div>
        </div>
      </div>

      <Section n="2" label="Fusion / Deep Hybrid Models" icon={GitMerge} isDark={isDark} />
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-3">
        <ModelCard name="Late Fusion (ALS + SBERT)" type="Hybrid (linear)" tone="amber" isDark={isDark}
          note="α clamp ≤ 0.8 ở app. α=1.0 ⇒ fused = CF thuần (KHÔNG fusion). Full-test best α=1.0 là degenerate ≡ ALS.">
          <Chip text="input: cf_score, sbert_score, α ∈ [0,1]" tone="amber" isDark={isDark} />
          <Formula text="fused = α · CF_score + (1−α) · SBERT_score" isDark={isDark} highlight />
          <Chip text="rank candidates desc by fused → Top-K" tone="amber" isDark={isDark} />
        </ModelCard>

        <ModelCard name="Light Gated Late Fusion" type="Deep Hybrid" tone="amber" recommended isDark={isDark}
          note="Gate học trọng số fusion theo từng (u,i). Train: 1 pos + 4 neg, BCE/BCEWithLogits, Adam; eval batched. Best deep hybrid (full test).">
          <Chip text="u_cf=ALS user_f(32) · i_cf=ALS item_f(32) · i_sbert(768) · scores" tone="amber" isDark={isDark} />
          <Chip text="feat = concat([u_cf, i_cf, i_sbert]) → 832-d" tone="amber" isDark={isDark} />
          <div className={`rounded-lg border p-2 ${isDark ? 'border-amber-500/30 bg-slate-900/40' : 'border-amber-200 bg-white/70'}`}>
            <div className={`text-[10px] font-bold mb-1.5 flex items-center gap-1 ${isDark ? 'text-amber-300' : 'text-amber-700'}`}><Zap className="h-3 w-3" />Gate MLP (per user,item)</div>
            <div className="space-y-1">
              <Chip text="Linear(832→128) → ReLU → Dropout(0.1)" tone="slate" isDark={isDark} />
              <Chip text="Linear(128→64)  → ReLU → Dropout(0.1)" tone="slate" isDark={isDark} />
              <Chip text="Linear(64→1)    → Sigmoid ⇒ α(u,i)" tone="slate" isDark={isDark} />
            </div>
          </div>
          <Formula text="fused = α(u,i)·CF + (1−α(u,i))·SBERT" isDark={isDark} highlight />
        </ModelCard>

        <ModelCard name="Deep Early Fusion (DCN)" type="Deep Hybrid" tone="amber" isDark={isDark}
          note="Early fusion: concat feature, học tương tác chéo (cross) + phi tuyến (deep). Output là điểm xếp hạng.">
          <Chip text="x = concat([user_cf 32, item_cf 32, item_sbert 768]) → 832-d" tone="amber" isDark={isDark} />
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            <div className={`rounded-lg border p-2 ${isDark ? 'border-amber-500/30 bg-slate-900/40' : 'border-amber-200 bg-white/70'}`}>
              <div className={`text-[10px] font-bold mb-1.5 ${isDark ? 'text-amber-300' : 'text-amber-700'}`}>Cross Network (×4)</div>
              <div className="space-y-1">
                <Chip text="x0 = x" tone="slate" isDark={isDark} />
                <Chip text="x_l = x0 ⊙ σ(W_l·x_{l-1}) + x_{l-1}" tone="slate" isDark={isDark} />
                <Chip text="cross_out = mean(x_4, dim=-1)" tone="slate" isDark={isDark} />
              </div>
            </div>
            <div className={`rounded-lg border p-2 ${isDark ? 'border-amber-500/30 bg-slate-900/40' : 'border-amber-200 bg-white/70'}`}>
              <div className={`text-[10px] font-bold mb-1.5 ${isDark ? 'text-amber-300' : 'text-amber-700'}`}>Deep MLP</div>
              <div className="space-y-1">
                <Chip text="Linear(832→256) → ReLU" tone="slate" isDark={isDark} />
                <Chip text="Linear(256→256) → ReLU" tone="slate" isDark={isDark} />
                <Chip text="Linear(256→128) → ReLU" tone="slate" isDark={isDark} />
                <Chip text="Linear(128→1) ⇒ deep_out" tone="slate" isDark={isDark} />
              </div>
            </div>
          </div>
          <Formula text="score = cross_out + deep_out" isDark={isDark} highlight />
        </ModelCard>

        <ModelCard name="Two-Tower (Retrieval / Ranking)" type="Deep Hybrid" tone="amber" isDark={isDark}
          note="Dùng LightGCN graph factors. Item tower kết hợp graph + SBERT; similarity = dot product (out dim 64).">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            <div className={`rounded-lg border p-2 ${isDark ? 'border-blue-500/30 bg-slate-900/40' : 'border-blue-200 bg-white/70'}`}>
              <div className={`text-[10px] font-bold mb-1.5 ${isDark ? 'text-blue-300' : 'text-blue-700'}`}>User Tower</div>
              <div className="space-y-1">
                <Chip text="in: graph_user[u] (32)" tone="blue" isDark={isDark} />
                <Chip text="Linear(32→128) → ReLU → Dropout(0.1)" tone="slate" isDark={isDark} />
                <Chip text="Linear(128→64) ⇒ u_emb (64)" tone="slate" isDark={isDark} />
              </div>
            </div>
            <div className={`rounded-lg border p-2 ${isDark ? 'border-emerald-500/30 bg-slate-900/40' : 'border-emerald-200 bg-white/70'}`}>
              <div className={`text-[10px] font-bold mb-1.5 ${isDark ? 'text-emerald-300' : 'text-emerald-700'}`}>Item Tower</div>
              <div className="space-y-1">
                <Chip text="in: concat(graph_item 32, sbert 768)=800" tone="emerald" isDark={isDark} />
                <Chip text="Linear(800→256) → ReLU → Dropout(0.1)" tone="slate" isDark={isDark} />
                <Chip text="Linear(256→256) → ReLU → Dropout(0.1)" tone="slate" isDark={isDark} />
                <Chip text="Linear(256→64) ⇒ i_emb (64)" tone="slate" isDark={isDark} />
              </div>
            </div>
          </div>
          <Formula text="score(u,i) = u_emb · i_emb" isDark={isDark} highlight />
        </ModelCard>

        <ModelCard name="Transformer Fusion (candidates)" type="Deep Hybrid" tone="amber" isDark={isDark}
          note="Self-attention trên tập candidate của 1 user để mô hình hóa tương tác giữa ứng viên; batch_first.">
          <Chip text="feat_c = concat(graph_user 32, graph_item 32, sbert 768) → 832" tone="amber" isDark={isDark} />
          <Chip text="sequence: (B, C, 832) cho C candidates / user" tone="amber" isDark={isDark} />
          <div className={`rounded-lg border p-2 ${isDark ? 'border-amber-500/30 bg-slate-900/40' : 'border-amber-200 bg-white/70'}`}>
            <div className={`text-[10px] font-bold mb-1.5 flex items-center gap-1 ${isDark ? 'text-amber-300' : 'text-amber-700'}`}><Layers className="h-3 w-3" />Transformer Encoder</div>
            <div className="space-y-1">
              <Chip text="proj: Linear(832→128) ⇒ d_model=128" tone="slate" isDark={isDark} />
              <Chip text="EncoderLayer ×2, nhead=4, d_ff=512, GELU, drop=0.1" tone="slate" isDark={isDark} />
              <Chip text="head: Linear(128→1) ⇒ score / candidate" tone="slate" isDark={isDark} />
            </div>
          </div>
          <Formula text="scores = head(Encoder(proj(X))) → rank desc" isDark={isDark} highlight />
        </ModelCard>

        <ModelCard name="Triển khai / Serving" type="Info" tone="slate" isDark={isDark}
          note="Graph Light Gated (kernel) tương tự Light Gated nhưng dùng LightGCN factors. Training ref: 1 pos+4 neg, Adam/BCE, 20–30 epochs (tuỳ model).">
          <Chip text="Mmap float16; infer head input dim từ checkpoint (384/768-d SBERT, 448/832-d head)." tone="slate" isDark={isDark} />
          <Chip text="Pipeline: resolve user → CF(ALS/BPR)+SBERT(FAISS/profile) → union candidates → fusion → Top-K + SQLite metadata." tone="slate" isDark={isDark} />
          <Chip text="API: /recommend, /simulate, /metrics, /health, /models. Alpha clamp ≤ 0.8; α=1 = pure CF." tone="slate" isDark={isDark} />
        </ModelCard>
      </div>

      <Section n="3" label="Đầu ra & đánh giá" icon={BarChart3} isDark={isDark} />
      <div className="grid grid-cols-1 md:grid-cols-[1fr_auto_1fr] items-center gap-2.5">
        <MiniBox title="Xếp hạng Top-K" sub="re-rank theo fused/model score (desc) → trả k items" tone="rose" isDark={isDark} wide />
        <div className={`hidden md:flex items-center justify-center text-xs font-mono ${sub}`}>→ evaluate →</div>
        <MiniBox title="Metrics (full test)" sub="HR@5/10/20 · NDCG@5/10/20 · MRR · Recall@5/10/20" tone="rose" isDark={isDark} wide />
      </div>

      <div className={`mt-4 grid grid-cols-1 md:grid-cols-3 gap-2.5 text-[10px] ${sub}`}>
        <div className={`rounded-xl border p-3 ${isDark ? 'border-slate-700 bg-slate-800/40' : 'border-slate-200 bg-slate-50/70'}`}>
          <div className={`font-semibold mb-1 ${isDark ? 'text-slate-200' : 'text-slate-700'}`}>Protocol</div>
          1 positive + 99 negatives / user, shared 200k catalog, leak-free SBERT, <strong className={isDark ? 'text-slate-100' : 'text-slate-800'}>299,455</strong> eligible users (min_interactions=3). Full test set.
        </div>
        <div className={`rounded-xl border p-3 ${isDark ? 'border-slate-700 bg-slate-800/40' : 'border-slate-200 bg-slate-50/70'}`}>
          <div className={`font-semibold mb-1 ${isDark ? 'text-slate-200' : 'text-slate-700'}`}>Baselines (full test)</div>
          ALS HR@10 <strong className={isDark ? 'text-slate-100' : 'text-slate-800'}>0.6153</strong> (best overall). <strong className={isDark ? 'text-indigo-300' : 'text-indigo-700'}>Light Gated HR@10 0.5245</strong> (best deep hybrid). Late Fusion best α=1.0 ≡ ALS (degenerate).
        </div>
        <div className={`rounded-xl border p-3 ${isDark ? 'border-slate-700 bg-slate-800/40' : 'border-slate-200 bg-slate-50/70'}`}>
          <div className={`font-semibold mb-1 ${isDark ? 'text-slate-200' : 'text-slate-700'}`}>Artifacts</div>
          <span className="font-mono">factors/*.npy(f16)</span>, <span className="font-mono">sbert_item_embeddings_f16.npy</span> (384-d local / 768-d target), <span className="font-mono">models/*.pt</span>, FAISS index.
        </div>
      </div>
    </div>
  );
}
