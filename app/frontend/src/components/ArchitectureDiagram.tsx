import React from 'react';

interface BoxProps {
  title: string;
  sub?: string;
  color: string;
  small?: boolean;
}

function Box({ title, sub, color, small }: BoxProps) {
  return (
    <div className={`rounded-lg border-2 ${color} ${small ? 'px-3 py-2' : 'px-4 py-3'} text-center shadow-sm`}>
      <div className={`font-bold ${small ? 'text-xs' : 'text-sm'}`}>{title}</div>
      {sub && <div className={`text-[10px] mt-0.5 opacity-70`}>{sub}</div>}
    </div>
  );
}

function Arrow({ vertical = false }: { vertical?: boolean }) {
  return vertical
    ? <div className="flex justify-center py-1"><div className="w-0.5 h-6 bg-slate-300" /><div className="w-0 h-0 border-l-4 border-r-4 border-t-6 border-l-transparent border-r-transparent border-t-slate-300 -mt-1" /></div>
    : <div className="flex items-center px-1"><div className="h-0.5 w-6 bg-slate-300" /><div className="w-0 h-0 border-t-4 border-b-4 border-l-6 border-t-transparent border-b-transparent border-l-slate-300 -ml-1" /></div>;
}

export default function ArchitectureDiagram({ isDark }: { isDark: boolean }) {
  const box = (title: string, sub: string, color: string) => (
    <div className={`rounded-lg border ${color} px-3 py-2 text-center`}>
      <div className="text-xs font-bold">{title}</div>
      <div className="text-[9px] opacity-60 mt-0.5">{sub}</div>
    </div>
  );

  const section = (label: string) => (
    <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400 mb-1 mt-3">{label}</div>
  );

  return (
    <div className={`rounded-xl border ${isDark ? 'border-slate-800 bg-slate-900' : 'border-slate-200 bg-white'} p-6`}>
      <h3 className="text-base font-bold mb-4">System Architecture — Deep Hybrid Recommender</h3>

      {/* Data layer */}
      {section('1. Data Layer')}
      <div className="grid grid-cols-3 gap-2">
        {box('Train Parquet', '28.1M interactions', 'border-blue-300 bg-blue-50')}
        {box('Test Parquet', '8.28M positives', 'border-blue-300 bg-blue-50')}
        {box('Item Profile', '516k real metadata', 'border-blue-300 bg-blue-50')}
      </div>
      <Arrow vertical />

      {/* CF layer */}
      {section('2. Collaborative Filtering (full 2.46M items)')}
      <div className="grid grid-cols-3 gap-2">
        {box('ALS (implicit)', '32-dim, min=3', 'border-purple-300 bg-purple-50')}
        {box('BPR (implicit)', '32-dim, min=3', 'border-purple-300 bg-purple-50')}
        {box('LightGCN', '32-dim, graph CF', 'border-purple-300 bg-purple-50')}
      </div>
      <Arrow vertical />

      {/* Content layer */}
      {section('3. Content-Based (SBERT, shared 200k catalog)')}
      <div className="grid grid-cols-2 gap-2">
        {box('SBERT Metadata', 'title+brand+cat+desc, 384-dim', 'border-emerald-300 bg-emerald-50')}
        {box('SBERT Reviewer', 'review_content, 384-dim', 'border-emerald-300 bg-emerald-50')}
      </div>
      <div className="text-center text-[10px] text-slate-400 my-1">↓ concatenate → 768-dim combined</div>
      <div className="flex justify-center">
        {box('FAISS IndexFlatIP', '200k items, cosine IP', 'border-emerald-300 bg-emerald-50')}
      </div>
      <Arrow vertical />

      {/* Fusion layer */}
      {section('4. Fusion / Deep Hybrid Models')}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
        {box('Late Fusion', 'α·CF + (1-α)·SBERT', 'border-amber-300 bg-amber-50')}
        {box('Light Gated', 'neural gate per (u,i)', 'border-amber-300 bg-amber-50')}
        {box('Deep Early (DCN)', 'cross-net + deep MLP', 'border-amber-300 bg-amber-50')}
        {box('Two-Tower', 'user/item towers, dot', 'border-amber-300 bg-amber-50')}
      </div>
      <div className="flex justify-center mt-2">
        {box('Transformer Fusion', 'self-attention over candidates', 'border-amber-300 bg-amber-50')}
      </div>
      <Arrow vertical />

      {/* Serving layer */}
      {section('5. Serving (8GB RAM, CPU)')}
      <div className="grid grid-cols-3 gap-2">
        {box('FastAPI', '/recommend, /simulate, /metrics', 'border-indigo-300 bg-indigo-50')}
        {box('Pipeline', 'mmap float16, top-k', 'border-indigo-300 bg-indigo-50')}
        {box('SQLite', 'real product metadata', 'border-indigo-300 bg-indigo-50')}
      </div>
      <Arrow vertical />

      {/* UI layer */}
      {section('6. Frontend (React + Vite)')}
      <div className="grid grid-cols-4 gap-2">
        {box('Storefront', 'product grid', 'border-rose-300 bg-rose-50', true)}
        {box('Benchmarks', 'leaderboard', 'border-rose-300 bg-rose-50', true)}
        {box('Simulator', 'A/B testing', 'border-rose-300 bg-rose-50', true)}
        {box('Architecture', 'this diagram', 'border-rose-300 bg-rose-50', true)}
      </div>
    </div>
  );
}
EOF