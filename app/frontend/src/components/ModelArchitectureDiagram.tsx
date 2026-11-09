import React from 'react';

function Box({ title, sub, color, wide }: { title: string; sub?: string; color: string; wide?: boolean }) {
  return (
    <div className={`rounded-lg border-2 ${color} ${wide ? 'px-6 py-3' : 'px-3 py-2'} text-center shadow-sm`}>
      <div className={`font-bold ${wide ? 'text-sm' : 'text-xs'}`}>{title}</div>
      {sub && <div className="text-[9px] mt-0.5 opacity-70">{sub}</div>}
    </div>
  );
}

function Arrow({ vertical = false, label }: { vertical?: boolean; label?: string }) {
  if (vertical) {
    return (
      <div className="flex flex-col items-center py-1">
        {label && <div className="text-[9px] text-slate-400 mb-0.5">{label}</div>}
        <div className="w-0.5 h-5 bg-slate-300" />
        <div className="w-0 h-0 border-l-4 border-r-4 border-t-6 border-l-transparent border-r-transparent border-t-slate-300" />
      </div>
    );
  }
  return (
    <div className="flex items-center px-1">
      {label && <div className="text-[9px] text-slate-400 mr-1">{label}</div>}
      <div className="h-0.5 w-5 bg-slate-300" />
      <div className="w-0 h-0 border-t-4 border-b-4 border-l-6 border-t-transparent border-b-transparent border-l-slate-300" />
    </div>
  );
}

function Section({ label }: { label: string }) {
  return <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400 mb-1 mt-4">{label}</div>;
}

export default function ModelArchitectureDiagram({ isDark }: { isDark: boolean }) {
  const card = isDark ? 'border-slate-700 bg-slate-900' : 'border-slate-200 bg-white';

  return (
    <div className={`rounded-xl border ${card} p-6`}>
      <h3 className="text-base font-bold mb-2">Model Architecture — Deep Hybrid Recommender</h3>
      <p className="text-xs text-slate-500 mb-4">From raw interactions to ranked recommendations</p>

      {/* Input */}
      {Section('1. Input')}
      <div className="flex justify-center gap-4">
        <Box title="User History" sub="item_idx list" color="border-blue-300 bg-blue-50" />
        <Box title="Item Catalog" sub="200k shared items" color="border-blue-300 bg-blue-50" />
      </div>
      <Arrow vertical />

      {/* CF branch */}
      {Section('2a. Collaborative Filtering Branch')}
      <div className="grid grid-cols-3 gap-2">
        <Box title="ALS (implicit)" sub="32-dim factors" color="border-purple-300 bg-purple-50" />
        <Box title="BPR (implicit)" sub="32-dim factors" color="border-purple-300 bg-purple-50" />
        <Box title="LightGCN" sub="32-dim graph emb" color="border-purple-300 bg-purple-50" />
      </div>
      <div className="text-center text-[10px] text-slate-400 my-1">user_emb · item_emb → CF score = item_emb · user_emb</div>
      <Arrow vertical label="CF score" />

      {/* Content branch */}
      {Section('2b. Content-Based Branch (SBERT)')}
      <div className="grid grid-cols-2 gap-2">
        <Box title="SBERT Metadata" sub="title+brand+cat+desc → 384-dim" color="border-emerald-300 bg-emerald-50" />
        <Box title="SBERT Reviewer" sub="review_content → 384-dim" color="border-emerald-300 bg-emerald-50" />
      </div>
      <div className="text-center text-[10px] text-slate-400 my-1">concatenate → 768-dim combined · FAISS IndexFlatIP (top-50)</div>
      <Arrow vertical label="Content score" />

      {/* Fusion */}
      {Section('3. Fusion / Deep Hybrid Models')}
      <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
        <Box title="Late Fusion" sub="α·CF + (1-α)·SBERT" color="border-amber-300 bg-amber-50" />
        <Box title="Light Gated" sub="neural gate per (u,i)" color="border-amber-300 bg-amber-50" />
        <Box title="Deep Early (DCN)" sub="cross-net + deep MLP" color="border-amber-300 bg-amber-50" />
      </div>
      <div className="grid grid-cols-2 gap-2 mt-2">
        <Box title="Two-Tower" sub="user tower · item tower" color="border-amber-300 bg-amber-50" />
        <Box title="Transformer Fusion" sub="self-attention over candidates" color="border-amber-300 bg-amber-50" />
      </div>
      <Arrow vertical label="fused scores" />

      {/* Output */}
      {Section('4. Output')}
      <div className="flex justify-center">
        <Box title="Ranked Top-K" sub="HR@K, NDCG@K, MRR" color="border-rose-300 bg-rose-50" wide />
      </div>

      {/* Training info */}
      <div className="mt-4 pt-3 border-t border-slate-200 text-[10px] text-slate-500 space-y-1">
        <div><strong>Training:</strong> CF on full 28.1M interactions; Deep hybrids on 50k-100k users (1 pos + 4 negs); 20-30 epochs, Adam, BCE loss</div>
        <div><strong>Eval:</strong> 1 positive + 99 random negatives, shared 200k catalog, leak-free SBERT profiles, 299,455 eligible users</div>
        <div><strong>Best model:</strong> Light Gated Late Fusion (HR@10 = 0.5245 on full test) — deep hybrid is the recommended model</div>
      </div>
    </div>
  );
}