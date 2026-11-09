import React, { useState, useEffect, useMemo } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line
} from 'recharts';
import {
  Sparkles, Database, User, Sun, Moon, Zap, BarChart3, Layers, Sliders,
  Award, CheckCircle2, Clock, RotateCcw, Play, ArrowUpDown, Search, ShoppingCart,
  Heart, Filter, ChevronRight, Package, TrendingUp, Cpu, HardDrive, ShieldCheck,
  ArrowUpRight, RefreshCw, Tag, ExternalLink, ThumbsUp, Plus, Info, Check
} from 'lucide-react';
import { BENCHMARK_METRICS, PENDING_MODELS } from './data/benchmark';
import { fetchSampleUsers, fetchRecommendations, fetchMetrics, runSimulation } from './services/api';
import type { Product, SampleUser, ResolvedUser, SimResult } from './types';

const MODELS = [
  { id: 'light_gated', name: 'Light Gated Late Fusion', type: 'Deep Hybrid', desc: 'Neural gate learns a per-(user,item) fusion weight' },
  { id: 'deep_early', name: 'Deep Early Fusion (DCN)', type: 'Deep Hybrid', desc: 'Cross-network over [user_cf, item_cf, item_sbert]' },
  { id: 'two_stage', name: 'Two-Stage Retrieval', type: 'Deep Hybrid', desc: 'CF + CB Candidate Union -> Reranking' },
  { id: 'als', name: 'Alternating Least Squares (ALS)', type: 'Collaborative', desc: 'Implicit matrix factorization (factors=32, min_interactions=3)' },
  { id: 'bpr', name: 'Bayesian Personalized Ranking (BPR)', type: 'Collaborative', desc: 'Pairwise ranking optimization (factors=32, min_interactions=3)' },
  { id: 'sbert', name: 'SBERT Content-Based', type: 'Content', desc: 'all-MiniLM-L6-v2 + FAISS IndexFlatIP' },
  { id: 'late_fusion', name: 'Late Fusion (ALS + SBERT)', type: 'Hybrid', desc: 'Linear combination with dynamic weight alpha' },
];

export default function App() {
  const [theme, setTheme] = useState<'dark' | 'light'>('light');
  const [activeTab, setActiveTab] = useState<'store' | 'leaderboard' | 'simulator' | 'architecture'>('store');
  const [customerId, setCustomerId] = useState<string>('A37OJC78HSOXXN');
  const [model, setModel] = useState<string>('light_gated');
  const [alpha, setAlpha] = useState<number>(0.65);
  const [topK, setTopK] = useState<number>(10);
  const [itemsMeta, setItemsMeta] = useState<Product[]>([]);
  const [latencyMs, setLatencyMs] = useState<number>(0);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [sampleUsers, setSampleUsers] = useState<SampleUser[]>([]);
  const [resolvedUser, setResolvedUser] = useState<ResolvedUser | null>(null);
  const [userHistory, setUserHistory] = useState<number[]>([]);

  // Simulator
  const [simUsers, setSimUsers] = useState<number>(2000);
  const [simNegs, setSimNegs] = useState<number>(99);
  const [isSimulating, setIsSimulating] = useState<boolean>(false);
  const [simProgress, setSimProgress] = useState<number>(0);
  const [simResults, setSimResults] = useState<SimResult | null>(null);
  const [simError, setSimError] = useState<string | null>(null);

  // Leaderboard
  const [sortField, setSortField] = useState<string>('hr10');
  const [sortAsc, setSortAsc] = useState<boolean>(false);

  const isDark = theme === 'dark';
  const bg = isDark ? 'bg-slate-950 text-slate-100' : 'bg-slate-50 text-slate-900';
  const card = isDark ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200';
  const muted = isDark ? 'text-slate-400' : 'text-slate-500';
  const border = isDark ? 'border-slate-800' : 'border-slate-200';

  useEffect(() => {
    fetchSampleUsers(12).then(d => {
      if (d.users?.length) {
        setSampleUsers(d.users);
        setCustomerId(d.users[0].customer_id);
      }
    }).catch(() => {});
  }, []);

  const fetchRecs = async (cid = customerId, mdl = model, a = alpha) => {
    setIsLoading(true);
    try {
      const data = await fetchRecommendations({ customer_id: cid, model: mdl, k: topK, alpha: a, history: userHistory });
      setItemsMeta(data.items_meta || []);
      setLatencyMs(data.latency_ms || 0);
      if (data.customer_id) setResolvedUser({ customer_id: data.customer_id, user_idx: data.user_idx, user_status: data.user_status || 'Verified' });
    } catch (e) { console.error(e); }
    finally { setIsLoading(false); }
  };

  useEffect(() => { fetchRecs(customerId, model, alpha); }, [customerId, model, topK]);

  const runSim = async () => {
    setIsSimulating(true);
    setSimError(null);
    setSimResults(null);
    setSimProgress(5);
    const t = setInterval(() => setSimProgress(p => (p < 90 ? p + 3 : p)), 200);
    try {
      const data = await runSimulation({ users: simUsers, negatives: simNegs, alpha });
      setSimResults(data);
      setSimProgress(100);
    } catch (e: any) {
      setSimError(e?.message || 'Failed');
      setSimProgress(0);
    } finally {
      clearInterval(t);
      setIsSimulating(false);
    }
  };

  const sortedLeaderboard = useMemo(() => {
    return [...BENCHMARK_METRICS].sort((a: any, b: any) => sortAsc ? a[sortField] - b[sortField] : b[sortField] - a[sortField]);
  }, [sortField, sortAsc]);

  const handleSort = (field: string) => {
    if (sortField === field) setSortAsc(!sortAsc);
    else { setSortField(field); setSortAsc(false); }
  };

  return (
    <div className={`min-h-screen ${bg}`}>
      {/* Header */}
      <header className={`sticky top-0 z-30 border-b ${isDark ? 'bg-slate-950/90 border-slate-800' : 'bg-white/90 border-slate-200'} backdrop-blur`}>
        <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between gap-4">
          <div className="flex items-center gap-2 font-extrabold text-lg tracking-tight">
            <Package className="h-6 w-6 text-indigo-600" />
            <span>HYBRID.<span className="text-indigo-600">REC</span></span>
          </div>
          <div className={`hidden md:flex flex-1 max-w-md items-center gap-2 px-3 py-2 rounded-lg border ${border} ${isDark ? 'bg-slate-900' : 'bg-slate-100'}`}>
            <Search className="h-4 w-4 text-slate-400" />
            <input className="flex-1 bg-transparent outline-none text-sm" placeholder="Search products..." />
          </div>
          <div className="flex items-center gap-2">
            <button className={`p-2 rounded-lg border ${border}`}><Heart className="h-5 w-5" /></button>
            <button className={`p-2 rounded-lg border ${border} relative`}>
              <ShoppingCart className="h-5 w-5" />
              <span className="absolute -top-1 -right-1 h-4 w-4 rounded-full bg-indigo-600 text-white text-[10px] flex items-center justify-center">3</span>
            </button>
            <button onClick={() => setTheme(isDark ? 'light' : 'dark')} className={`p-2 rounded-lg border ${border}`}>
              {isDark ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
            </button>
          </div>
        </div>
        <nav className={`border-t ${border}`}>
          <div className="max-w-7xl mx-auto px-4 flex gap-1">
            {[
              { id: 'store', label: 'Storefront' },
              { id: 'leaderboard', label: 'Benchmarks' },
              { id: 'simulator', label: 'A/B Simulator' },
              { id: 'architecture', label: 'Architecture' },
            ].map(t => (
              <button key={t.id} onClick={() => setActiveTab(t.id as any)}
                className={`py-3 px-4 text-sm font-medium border-b-2 transition-colors ${
                  activeTab === t.id ? 'border-indigo-600 text-indigo-600' : 'border-transparent text-slate-500 hover:text-slate-900'
                }`}>
                {t.label}
              </button>
            ))}
          </div>
        </nav>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-6">
        {activeTab === 'store' && (
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
            {/* Sidebar */}
            <aside className="lg:col-span-1 space-y-4">
              <div className={`${card} border rounded-xl p-4`}>
                <h3 className="text-sm font-bold mb-3">Recommendation Studio</h3>
                <label className="text-xs font-medium text-slate-500 block mb-1">Customer ID</label>
                <input value={customerId} onChange={e => setCustomerId(e.target.value)}
                  className={`w-full px-3 py-2 rounded-lg border ${border} ${isDark ? 'bg-slate-800' : 'bg-slate-50'} text-sm mb-3`} />
                <label className="text-xs font-medium text-slate-500 block mb-1">Model</label>
                <select value={model} onChange={e => setModel(e.target.value)}
                  className={`w-full px-3 py-2 rounded-lg border ${border} ${isDark ? 'bg-slate-800' : 'bg-slate-50'} text-sm mb-3`}>
                  {MODELS.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}
                </select>
                <label className="text-xs font-medium text-slate-500 block mb-1">Alpha (CF weight): {Math.round(alpha * 100)}%</label>
                <input type="range" min="0" max="0.8" step="0.05" value={alpha}
                  onChange={e => { const v = parseFloat(e.target.value); setAlpha(v); fetchRecs(customerId, model, v); }}
                  className="w-full mb-3" />
                <label className="text-xs font-medium text-slate-500 block mb-1">Top K: {topK}</label>
                <input type="range" min="4" max="20" step="2" value={topK}
                  onChange={e => setTopK(Number(e.target.value))}
                  className="w-full mb-3" />
                <button onClick={() => fetchRecs(customerId, model, alpha)}
                  className="w-full py-2 rounded-lg bg-indigo-600 text-white text-sm font-medium hover:bg-indigo-700 transition-colors">
                  {isLoading ? 'Loading...' : 'Get Recommendations'}
                </button>
                {resolvedUser && (
                  <div className={`mt-3 text-xs ${muted}`}>
                    <div>User: {resolvedUser.customer_id}</div>
                    <div>Status: {resolvedUser.user_status}</div>
                    <div>Latency: {latencyMs}ms</div>
                  </div>
                )}
              </div>
              {/* Sample users */}
              <div className={`${card} border rounded-xl p-4`}>
                <h3 className="text-sm font-bold mb-2">Sample Users</h3>
                <div className="space-y-1">
                  {sampleUsers.slice(0, 6).map((u, i) => (
                    <button key={i} onClick={() => setCustomerId(u.customer_id)}
                      className={`w-full text-left text-xs px-2 py-1.5 rounded hover:bg-slate-100 ${isDark ? 'hover:bg-slate-800' : ''}`}>
                      {u.label || u.customer_id}
                    </button>
                  ))}
                </div>
              </div>
            </aside>

            {/* Product grid */}
            <section className="lg:col-span-3">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-bold">Recommended for You</h2>
                <span className={`text-sm ${muted}`}>{itemsMeta.length} items</span>
              </div>
              {isLoading ? (
                <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-4 gap-4">
                  {Array.from({ length: 8 }).map((_, i) => (
                    <div key={i} className={`${card} border rounded-xl p-4 animate-pulse`}>
                      <div className="h-32 bg-slate-200 rounded-lg mb-3" />
                      <div className="h-4 bg-slate-200 rounded mb-2" />
                      <div className="h-3 bg-slate-200 rounded w-2/3" />
                    </div>
                  ))}
                </div>
              ) : (
                <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-4 gap-4">
                  {itemsMeta.map((item, idx) => (
                    <div key={item.item_idx} className={`${card} border rounded-xl overflow-hidden hover:shadow-md transition-shadow`}>
                      <div className={`h-36 ${isDark ? 'bg-slate-800' : 'bg-slate-100'} flex items-center justify-center`}>
                        <Package className="h-12 w-12 text-slate-300" />
                      </div>
                      <div className="p-3">
                        <div className="text-[10px] font-medium text-indigo-600 uppercase tracking-wide mb-1">{item.category || 'Product'}</div>
                        <h4 className="text-sm font-semibold line-clamp-2 mb-1">{item.title}</h4>
                        {item.brand && <div className="text-xs text-slate-500 mb-1">{item.brand}</div>}
                        <div className="flex items-center justify-between mt-2">
                          {item.price && <span className="font-bold text-emerald-600">{item.price}</span>}
                          <button className="text-xs px-2 py-1 rounded bg-indigo-600 text-white hover:bg-indigo-700">Add to Cart</button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </section>
          </div>
        )}

        {activeTab === 'leaderboard' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className={`${card} border rounded-xl p-4`}>
                <div className="text-xs text-slate-500 mb-1">Top Model</div>
                <div className="text-lg font-bold text-indigo-600">Light Gated Late Fusion</div>
                <div className="text-xs text-emerald-600 font-mono mt-1">HR@10: 0.5245 | NDCG@10: 0.3499</div>
              </div>
              <div className={`${card} border rounded-xl p-4`}>
                <div className="text-xs text-slate-500 mb-1">Eval Users</div>
                <div className="text-lg font-bold">299,455</div>
                <div className="text-xs text-slate-500 font-mono mt-1">Eligible test users</div>
              </div>
              <div className={`${card} border rounded-xl p-4`}>
                <div className="text-xs text-slate-500 mb-1">Catalog</div>
                <div className="text-lg font-bold">200,000</div>
                <div className="text-xs text-slate-500 font-mono mt-1">Shared items</div>
              </div>
              <div className={`${card} border rounded-xl p-4`}>
                <div className="text-xs text-slate-500 mb-1">Min Interactions</div>
                <div className="text-lg font-bold">≥ 3</div>
                <div className="text-xs text-slate-500 font-mono mt-1">Train threshold</div>
              </div>
            </div>

            <div className={`${card} border rounded-xl overflow-hidden`}>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className={`${isDark ? 'bg-slate-800' : 'bg-slate-100'} text-xs uppercase`}>
                    <tr>
                      <th className="py-3 px-4 text-left">Model</th>
                      <th className="py-3 px-4 text-left">Type</th>
                      <th className="py-3 px-4 cursor-pointer" onClick={() => handleSort('hr10')}>HR@10</th>
                      <th className="py-3 px-4">NDCG@10</th>
                      <th className="py-3 px-4">MRR</th>
                      <th className="py-3 px-4 text-right">Time</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200">
                    {sortedLeaderboard.map((row: any, idx: number) => (
                      <tr key={idx} className={row.best ? 'bg-indigo-50' : ''}>
                        <td className="py-3 px-4 font-medium">{row.model} {row.best && <span className="text-xs bg-indigo-100 text-indigo-700 px-1.5 py-0.5 rounded ml-1">Top</span>}</td>
                        <td className="py-3 px-4 text-slate-500 text-xs">{row.type}</td>
                        <td className="py-3 px-4 font-bold text-indigo-600">{row.hr10.toFixed(4)}</td>
                        <td className="py-3 px-4 text-emerald-600">{row.ndcg10.toFixed(4)}</td>
                        <td className="py-3 px-4 text-pink-600">{row.mrr.toFixed(4)}</td>
                        <td className="py-3 px-4 text-right text-slate-400">{row.timeSec}s</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {PENDING_MODELS.length > 0 && (
              <div className={`${card} border rounded-xl p-4`}>
                <h3 className="text-sm font-bold mb-2">Pending Models</h3>
                {PENDING_MODELS.map((m: any, i: number) => (
                  <div key={i} className="text-xs text-slate-500 mb-1">{m.model} — {m.status}</div>
                ))}
              </div>
            )}
          </div>
        )}

        {activeTab === 'simulator' && (
          <div className="max-w-3xl mx-auto space-y-6">
            <div className={`${card} border rounded-xl p-6`}>
              <h3 className="text-lg font-bold mb-2">A/B Testing Simulator</h3>
              <p className="text-sm text-slate-500 mb-4">Simulate Leave-One-Out evaluation: 1 positive + N negatives per user.</p>
              <div className="grid grid-cols-2 gap-4 mb-4">
                <div>
                  <label className="text-xs font-medium text-slate-500 block mb-1">Users: {simUsers}</label>
                  <input type="range" min="500" max="5000" step="500" value={simUsers} onChange={e => setSimUsers(Number(e.target.value))} className="w-full" />
                </div>
                <div>
                  <label className="text-xs font-medium text-slate-500 block mb-1">Negatives: {simNegs}</label>
                  <input type="range" min="19" max="99" step="20" value={simNegs} onChange={e => setSimNegs(Number(e.target.value))} className="w-full" />
                </div>
              </div>
              <button onClick={runSim} disabled={isSimulating}
                className="px-5 py-2.5 bg-indigo-600 text-white text-sm font-medium rounded-lg hover:bg-indigo-700 disabled:opacity-50">
                {isSimulating ? 'Running...' : 'Run Simulation'}
              </button>
              {isSimulating && (
                <div className="mt-3">
                  <div className="w-full bg-slate-200 h-2 rounded-full overflow-hidden">
                    <div className="bg-indigo-600 h-full transition-all" style={{ width: `${simProgress}%` }} />
                  </div>
                  <div className="text-right text-xs text-slate-500 mt-1">{simProgress}%</div>
                </div>
              )}
              {simError && <div className="mt-3 text-sm text-red-600">{simError}</div>}
              {simResults && !isSimulating && (
                <div className="mt-4">
                  <table className="w-full text-sm">
                    <thead className="border-b">
                      <tr>
                        <th className="py-2 px-3 text-left">Model</th>
                        <th className="py-2 px-3">HR@5</th>
                        <th className="py-2 px-3">HR@10</th>
                        <th className="py-2 px-3">NDCG@10</th>
                        <th className="py-2 px-3">MRR</th>
                      </tr>
                    </thead>
                    <tbody>
                      {[...simResults.models].sort((a: any, b: any) => b.hr10 - a.hr10).map((row: any, idx: number) => (
                        <tr key={idx} className={idx === 0 ? 'bg-indigo-50' : ''}>
                          <td className="py-2 px-3 font-medium">{row.name} {idx === 0 && '🏆'}</td>
                          <td className="py-2 px-3">{row.hr5.toFixed(4)}</td>
                          <td className="py-2 px-3 font-bold text-indigo-600">{row.hr10.toFixed(4)}</td>
                          <td className="py-2 px-3 text-emerald-600">{row.ndcg10.toFixed(4)}</td>
                          <td className="py-2 px-3 text-pink-600">{row.mrr.toFixed(4)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'architecture' && (
          <div className="max-w-5xl mx-auto">
            <ArchitectureDiagram isDark={isDark} />
          </div>
        )}
      </main>

      <footer className={`border-t py-4 text-center text-xs ${muted}`}>
        HYBRID.REC © 2026 — Deep Hybrid Recommender System. 8GB RAM CPU Optimized.
      </footer>
    </div>
  );
}