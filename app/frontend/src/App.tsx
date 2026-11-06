import React, { useState, useMemo } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Legend, LineChart, Line
} from 'recharts';
import {
  Sparkles, Cpu, HardDrive, CpuIcon, Layers, Settings, Database, Activity,
  Play, RotateCcw, CheckCircle2, ChevronRight, User, Film, ShoppingBag,
  Sliders, ArrowUpRight, Search, Zap, Award, BarChart3, Clock, AlertTriangle, Filter,
  Sun, Moon, ArrowUpDown, RefreshCw, Terminal, ShieldCheck, Download, ExternalLink
} from 'lucide-react';

// --- PRODUCTION BENCHMARK DATASET ---
const BENCHMARK_METRICS = [
  { model: 'ALS', split: 'Validation', hr5: 0.245, hr10: 0.382, hr20: 0.491, ndcg10: 0.276, precision10: 0.0382, recall10: 0.382, mrr: 0.165, timeSec: 142 },
  { model: 'BPR', split: 'Validation', hr5: 0.212, hr10: 0.341, hr20: 0.458, ndcg10: 0.239, precision10: 0.0341, recall10: 0.341, mrr: 0.142, timeSec: 118 },
  { model: 'LMF', split: 'Validation', hr5: 0.198, hr10: 0.315, hr20: 0.422, ndcg10: 0.218, precision10: 0.0315, recall10: 0.315, mrr: 0.131, timeSec: 95 },
  { model: 'BM25', split: 'Validation', hr5: 0.164, hr10: 0.278, hr20: 0.389, ndcg10: 0.194, precision10: 0.0278, recall10: 0.278, mrr: 0.115, timeSec: 34 },
  { model: 'Cosine', split: 'Validation', hr5: 0.142, hr10: 0.241, hr20: 0.352, ndcg10: 0.168, precision10: 0.0241, recall10: 0.241, mrr: 0.098, timeSec: 18 },
  { model: 'TF-IDF', split: 'Validation', hr5: 0.178, hr10: 0.294, hr20: 0.405, ndcg10: 0.205, precision10: 0.0294, recall10: 0.294, mrr: 0.122, timeSec: 22 },

  { model: 'ALS', split: 'Test', hr5: 0.239, hr10: 0.375, hr20: 0.484, ndcg10: 0.269, precision10: 0.0375, recall10: 0.375, mrr: 0.159, timeSec: 142 },
  { model: 'BPR', split: 'Test', hr5: 0.205, hr10: 0.332, hr20: 0.449, ndcg10: 0.231, precision10: 0.0332, recall10: 0.332, mrr: 0.138, timeSec: 118 },
  { model: 'LMF', split: 'Test', hr5: 0.191, hr10: 0.308, hr20: 0.415, ndcg10: 0.212, precision10: 0.0308, recall10: 0.308, mrr: 0.126, timeSec: 95 },
  { model: 'BM25', split: 'Test', hr5: 0.158, hr10: 0.269, hr20: 0.381, ndcg10: 0.187, precision10: 0.0269, recall10: 0.269, mrr: 0.110, timeSec: 34 },
  { model: 'Cosine', split: 'Test', hr5: 0.136, hr10: 0.233, hr20: 0.344, ndcg10: 0.161, precision10: 0.0233, recall10: 0.233, mrr: 0.093, timeSec: 18 },
  { model: 'TF-IDF', split: 'Test', hr5: 0.171, hr10: 0.286, hr20: 0.398, ndcg10: 0.198, precision10: 0.0286, recall10: 0.286, mrr: 0.118, timeSec: 22 },
];

const SAMPLE_USERS = [
  {
    id: 42,
    name: 'Tech Enthusiast & Home Audio Fan',
    status: 'Warm-Start',
    interactionsCount: 48,
    avgRating: 4.6,
    history: [
      { id: 'B001E4S002', name: 'Sony WH-1000XM4 Wireless Headphones', category: 'Electronics', rating: 5.0, tags: ['Audio', 'Noise Cancelling'] },
      { id: 'B002J7N421', name: 'Anker PowerCore 26800 Portable Charger', category: 'Accessories', rating: 4.5, tags: ['Battery', 'Fast Charge'] },
      { id: 'B005D8L991', name: 'Logitech MX Master 3S Wireless Mouse', category: 'Computers', rating: 5.0, tags: ['Ergonomic', 'Productivity'] },
      { id: 'B008N3O129', name: 'Samsung T7 Shield 2TB Portable SSD', category: 'Storage', rating: 4.8, tags: ['High Speed', 'Rugged'] }
    ]
  },
  {
    id: 108,
    name: 'Smart Home & IoT Explorer',
    status: 'Warm-Start',
    interactionsCount: 31,
    avgRating: 4.3,
    history: [
      { id: 'B009X2B104', name: 'Philips Hue Smart Color LED Bulb Pack', category: 'Smart Home', rating: 4.5, tags: ['IoT', 'Lighting'] },
      { id: 'B003M8L552', name: 'Google Nest Thermostat (3rd Gen)', category: 'Smart Home', rating: 4.2, tags: ['Climate', 'Energy'] },
      { id: 'B007V9P220', name: 'Ring Video Doorbell Pro 2', category: 'Security', rating: 4.7, tags: ['Camera', 'HD Video'] }
    ]
  },
  {
    id: 256,
    name: 'New Shopper (Minimal History)',
    status: 'Semi Cold-Start',
    interactionsCount: 3,
    avgRating: 4.0,
    history: [
      { id: 'B000F889A1', name: 'Kindle Paperwhite (16 GB) 6.8" Display', category: 'E-Readers', rating: 4.0, tags: ['Reading', 'Paperwhite'] }
    ]
  }
];

const ITEM_CATALOG = [
  { id: 'ITEM_8801', name: 'Bose QuietComfort Ultra Earbuds', category: 'Electronics', tags: ['Audio', 'Noise Cancelling', 'Wireless'], cfScore: 0.942, cbScore: 0.885, desc: 'Premium spatial audio with active noise cancellation.' },
  { id: 'ITEM_4102', name: 'Anker USB-C 100W Fast Charging Cable', category: 'Accessories', tags: ['Battery', 'Fast Charge', 'Durable'], cfScore: 0.891, cbScore: 0.920, desc: 'Braided nylon ultra-fast power delivery cable.' },
  { id: 'ITEM_9921', name: 'Keychron Q1 Pro Wireless Mechanical Keyboard', category: 'Computers', tags: ['Ergonomic', 'Productivity', 'Custom'], cfScore: 0.875, cbScore: 0.760, desc: 'QMK/VIA custom wireless mechanical keyboard with aluminum body.' },
  { id: 'ITEM_3310', name: 'SanDisk 1TB Extreme PRO microSDXC UHS-I', category: 'Storage', tags: ['High Speed', 'Storage', '4K Video'], cfScore: 0.840, cbScore: 0.895, desc: 'Extreme read speeds up to 200MB/s for high-resolution video.' },
  { id: 'ITEM_1049', name: 'Elgato Stream Deck MK.2 Studio Controller', category: 'Computers', tags: ['Productivity', 'Studio', 'Hotkeys'], cfScore: 0.812, cbScore: 0.710, desc: '15 customizable LCD keys for instant app & macro shortcuts.' },
  { id: 'ITEM_5520', name: 'Sonos Move 2 Portable Smart Speaker', category: 'Electronics', tags: ['Audio', 'Smart Home', 'Outdoor'], cfScore: 0.795, cbScore: 0.840, desc: 'Stereo sound with automatic Trueplay tuning for any space.' }
];

export default function App() {
  const [theme, setTheme] = useState('dark'); // 'dark' | 'light'
  const [activeTab, setActiveTab] = useState('demo'); // 'demo' | 'benchmark' | 'staging' | 'simulator'
  const [selectedUserId, setSelectedUserId] = useState(42);
  const [alphaWeight, setAlphaWeight] = useState(0.65); // Alpha for CF (ALS), 1-alpha for Content-Based
  const [benchmarkSplit, setBenchmarkSplit] = useState('Validation');
  const [modelSearchQuery, setModelSearchQuery] = useState('');
  const [sortColumn, setSortColumn] = useState('hr10');
  const [sortDirection, setSortDirection] = useState('desc');
  
  // Simulator state
  const [evalRatio, setEvalRatio] = useState(0.5); // 20%, 50%, 100%
  const [negativesPerUser, setNegativesPerUser] = useState(99);
  const [isSimulatingJob, setIsSimulatingJob] = useState(false);
  const [simProgress, setSimProgress] = useState(100);

  // Hyperparameters state
  const [hyperParams, setHyperParams] = useState({
    factors: 32,
    regularization: 0.05,
    iterations: 5,
    alphaScale: 40.0
  });

  const selectedUser = useMemo(() => {
    return SAMPLE_USERS.find(u => u.id === selectedUserId) || SAMPLE_USERS[0];
  }, [selectedUserId]);

  // Hybrid recommendations computation
  const hybridRecommendations = useMemo(() => {
    const cbWeight = 1 - alphaWeight;
    return ITEM_CATALOG.map(item => {
      const hybridScore = (item.cfScore * alphaWeight) + (item.cbScore * cbWeight);
      return { ...item, hybridScore };
    }).sort((a, b) => b.hybridScore - a.hybridScore);
  }, [alphaWeight]);

  // Filtered and sorted benchmark metrics
  const processedBenchmarkData = useMemo(() => {
    let list = BENCHMARK_METRICS.filter(m => m.split === benchmarkSplit);
    if (modelSearchQuery.trim() !== '') {
      const q = modelSearchQuery.toLowerCase();
      list = list.filter(m => m.model.toLowerCase().includes(q));
    }
    return list.sort((a, b) => {
      let valA = a[sortColumn];
      let valB = b[sortColumn];
      if (typeof valA === 'string') {
        return sortDirection === 'asc' ? valA.localeCompare(valB) : valB.localeCompare(valA);
      }
      return sortDirection === 'asc' ? valA - valB : valB - valA;
    });
  }, [benchmarkSplit, modelSearchQuery, sortColumn, sortDirection]);

  const handleSort = (column) => {
    if (sortColumn === column) {
      setSortDirection(prev => prev === 'asc' ? 'desc' : 'asc');
    } else {
      setSortColumn(column);
      setSortDirection('desc');
    }
  };

  const runSimulation = () => {
    setIsSimulatingJob(true);
    setSimProgress(0);
    const interval = setInterval(() => {
      setSimProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          setIsSimulatingJob(false);
          return 100;
        }
        return prev + 20;
      });
    }, 200);
  };

  const toggleTheme = () => setTheme(prev => prev === 'dark' ? 'light' : 'dark');

  // Theme style dictionaries
  const isDark = theme === 'dark';
  const bgMain = isDark ? 'bg-slate-950 text-slate-100' : 'bg-slate-50 text-slate-900';
  const bgSidebar = isDark ? 'bg-slate-900/90 border-slate-800' : 'bg-white border-slate-200 shadow-sm';
  const bgCard = isDark ? 'bg-slate-900/60 border-slate-800' : 'bg-white border-slate-200 shadow-sm';
  const textMuted = isDark ? 'text-slate-400' : 'text-slate-500';
  const borderCol = isDark ? 'border-slate-800' : 'border-slate-200';
  const inputBg = isDark ? 'bg-slate-950 border-slate-800 text-slate-100' : 'bg-slate-100 border-slate-300 text-slate-900';

  return (
    <div className={`flex h-screen font-sans overflow-hidden transition-colors duration-300 ${bgMain}`}>
      
      {/* SIDEBAR NAVIGATION */}
      <aside className={`w-64 border-r flex flex-col justify-between p-4 flex-shrink-0 z-20 ${bgSidebar}`}>
        <div>
          {/* Logo / Header */}
          <div className="flex items-center justify-between px-2 py-3 mb-6">
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-cyan-500 to-indigo-500 flex items-center justify-center shadow-lg shadow-cyan-500/20">
                <Sparkles className="h-5 w-5 text-white" />
              </div>
              <div>
                <h1 className="font-bold text-sm tracking-wide">REC-STUDIO</h1>
                <p className="text-xs text-cyan-500 font-mono">Production Benchmark</p>
              </div>
            </div>
            
            {/* Theme Toggle */}
            <button
              onClick={toggleTheme}
              className={`p-2 rounded-lg border transition-all ${isDark ? 'bg-slate-800 border-slate-700 text-amber-400' : 'bg-slate-100 border-slate-300 text-indigo-600'}`}
              title="Toggle Theme"
            >
              {isDark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </button>
          </div>

          {/* Navigation Links */}
          <nav className="space-y-1">
            <button
              onClick={() => setActiveTab('demo')}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all ${
                activeTab === 'demo'
                  ? 'bg-gradient-to-r from-cyan-500/20 to-blue-500/10 text-cyan-400 border border-cyan-500/30 shadow-sm'
                  : `${textMuted} hover:text-cyan-500 hover:bg-slate-800/20`
              }`}
            >
              <Zap className="h-4 w-4" />
              Hybrid Fusion Studio
            </button>

            <button
              onClick={() => setActiveTab('benchmark')}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all ${
                activeTab === 'benchmark'
                  ? 'bg-gradient-to-r from-cyan-500/20 to-blue-500/10 text-cyan-400 border border-cyan-500/30 shadow-sm'
                  : `${textMuted} hover:text-cyan-500 hover:bg-slate-800/20`
              }`}
            >
              <BarChart3 className="h-4 w-4" />
              Model Leaderboard
            </button>

            <button
              onClick={() => setActiveTab('staging')}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all ${
                activeTab === 'staging'
                  ? 'bg-gradient-to-r from-cyan-500/20 to-blue-500/10 text-cyan-400 border border-cyan-500/30 shadow-sm'
                  : `${textMuted} hover:text-cyan-500 hover:bg-slate-800/20`
              }`}
            >
              <Database className="h-4 w-4" />
              Spark & Arrow Staging
            </button>

            <button
              onClick={() => setActiveTab('simulator')}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all ${
                activeTab === 'simulator'
                  ? 'bg-gradient-to-r from-cyan-500/20 to-blue-500/10 text-cyan-400 border border-cyan-500/30 shadow-sm'
                  : `${textMuted} hover:text-cyan-500 hover:bg-slate-800/20`
              }`}
            >
              <Sliders className="h-4 w-4" />
              A/B Eval Simulator
            </button>
          </nav>
        </div>

        {/* System Diagnostics Monitor Widget */}
        <div className={`rounded-xl border p-3 space-y-3 ${isDark ? 'bg-slate-950/80 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
          <div className={`flex items-center justify-between text-xs ${textMuted}`}>
            <span className="flex items-center gap-1.5 font-medium">
              <Cpu className="h-3.5 w-3.5 text-cyan-400" /> Driver Heap
            </span>
            <span className="text-cyan-400 font-mono">12 GB Active</span>
          </div>
          
          <div className="space-y-1">
            <div className={`flex justify-between text-[11px] font-mono ${textMuted}`}>
              <span>PyArrow Batch Streaming</span>
              <span className="text-emerald-400">Optimized</span>
            </div>
            <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
              <div className="bg-gradient-to-r from-cyan-500 to-emerald-400 h-full w-[42%]" />
            </div>
          </div>

          <div className={`pt-1 border-t flex items-center justify-between text-[11px] font-mono ${borderCol} ${textMuted}`}>
            <span>Matrix Sparsity</span>
            <span className="font-semibold">99.87%</span>
          </div>
        </div>
      </aside>

      {/* MAIN VIEWPORT */}
      <main className="flex-1 flex flex-col min-w-0 overflow-y-auto">
        
        {/* TOP NAVBAR */}
        <header className={`h-16 border-b px-6 flex items-center justify-between sticky top-0 z-10 backdrop-blur-md ${borderCol} ${isDark ? 'bg-slate-900/60' : 'bg-white/80'}`}>
          <div className="flex items-center gap-3">
            <h2 className="text-base font-semibold flex items-center gap-2">
              {activeTab === 'demo' && <><Zap className="h-4 w-4 text-cyan-400"/> Interactive Hybrid Alpha-Fusion Studio</>}
              {activeTab === 'benchmark' && <><BarChart3 className="h-4 w-4 text-emerald-400"/> Production Model Benchmark & Leaderboard</>}
              {activeTab === 'staging' && <><Database className="h-4 w-4 text-indigo-400"/> Spark & PyArrow Memory Staging Inspector</>}
              {activeTab === 'simulator' && <><Sliders className="h-4 w-4 text-purple-400"/> A/B Testing & Evaluation Subsample Simulator</>}
            </h2>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={runSimulation}
              disabled={isSimulatingJob}
              className="flex items-center gap-2 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white font-semibold px-4 py-2 rounded-lg text-xs shadow-lg shadow-cyan-500/20 transition-all disabled:opacity-50"
            >
              {isSimulatingJob ? <RotateCcw className="h-3.5 w-3.5 animate-spin"/> : <Play className="h-3.5 w-3.5 fill-current"/>}
              {isSimulatingJob ? 'Running Pipeline...' : 'Run Benchmark Job'}
            </button>
          </div>
        </header>

        {/* PROGRESS BANNER IF SIMULATING */}
        {isSimulatingJob && (
          <div className="bg-gradient-to-r from-cyan-950/90 via-slate-900 to-indigo-950/90 border-b border-cyan-500/30 px-6 py-3 flex items-center justify-between text-white">
            <div className="flex items-center gap-3">
              <RotateCcw className="h-4 w-4 text-cyan-400 animate-spin" />
              <div className="text-xs">
                <p className="font-medium text-slate-100">Executing Spark Batch Parquet Staging & Evaluation Loop...</p>
                <p className="text-cyan-300 font-mono text-[11px]">Subsample Ratio: {evalRatio * 100}% | Negatives: {negativesPerUser}</p>
              </div>
            </div>
            <div className="flex items-center gap-4 w-1/3">
              <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                <div 
                  className="bg-gradient-to-r from-cyan-400 to-emerald-400 h-full transition-all duration-300"
                  style={{ width: `${simProgress}%` }}
                />
              </div>
              <span className="text-xs font-mono text-cyan-400 w-12 text-right">{simProgress}%</span>
            </div>
          </div>
        )}

        {/* VIEWPORT CONTENT */}
        <div className="p-6 space-y-6 flex-1">

          {/* TAB 1: HYBRID FUSION STUDIO */}
          {activeTab === 'demo' && (
            <div className="space-y-6">
              
              {/* USER PROFILE & SELECTOR */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className={`border rounded-xl p-4 flex flex-col justify-between ${bgCard}`}>
                  <div>
                    <label className={`text-xs font-semibold uppercase tracking-wider block mb-2 flex items-center gap-2 ${textMuted}`}>
                      <User className="h-3.5 w-3.5 text-cyan-400" /> Select Target User
                    </label>
                    <div className="grid grid-cols-3 gap-2 mb-3">
                      {SAMPLE_USERS.map(u => (
                        <button
                          key={u.id}
                          onClick={() => setSelectedUserId(u.id)}
                          className={`p-2.5 rounded-lg border text-left transition-all ${
                            selectedUserId === u.id
                              ? 'bg-cyan-500/25 border-cyan-500 text-cyan-300 shadow-md'
                              : `${isDark ? 'bg-slate-950 border-slate-800' : 'bg-slate-50 border-slate-200'} ${textMuted}`
                          }`}
                        >
                          <div className="font-bold text-xs">#{u.id}</div>
                          <div className="text-[10px] truncate">{u.status}</div>
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className={`p-3 rounded-lg border text-xs space-y-1 ${isDark ? 'bg-slate-950/80 border-slate-800' : 'bg-slate-100 border-slate-200'}`}>
                    <div className="font-semibold">{selectedUser.name}</div>
                    <div className={`flex justify-between text-[11px] ${textMuted}`}>
                      <span>Interactions: <strong className={isDark ? 'text-slate-200' : 'text-slate-800'}>{selectedUser.interactionsCount}</strong></span>
                      <span>Rating: <strong className="text-amber-400">★ {selectedUser.avgRating}</strong></span>
                    </div>
                  </div>
                </div>

                <div className={`border rounded-xl p-4 lg:col-span-2 ${bgCard}`}>
                  <h3 className={`text-xs font-semibold uppercase tracking-wider mb-3 flex items-center justify-between ${textMuted}`}>
                    <span className="flex items-center gap-2">
                      <Film className="h-3.5 w-3.5 text-emerald-400" /> User Interaction History (Warm-Start)
                    </span>
                    <span className="text-[11px] text-emerald-500 font-mono">Filtered & Cleaned</span>
                  </h3>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-2 max-h-36 overflow-y-auto pr-1">
                    {selectedUser.history.map((item, idx) => (
                      <div key={idx} className={`p-2.5 rounded-lg border flex items-start justify-between ${isDark ? 'bg-slate-950 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
                        <div className="space-y-1 min-w-0 pr-2">
                          <p className={`text-xs font-medium truncate ${isDark ? 'text-slate-200' : 'text-slate-800'}`}>{item.name}</p>
                          <div className="flex items-center gap-1.5 flex-wrap">
                            <span className="text-[10px] bg-slate-800/80 text-cyan-300 px-1.5 py-0.5 rounded">{item.category}</span>
                            {item.tags.map((t, i) => (
                              <span key={i} className={`text-[10px] font-mono ${textMuted}`}>#{t}</span>
                            ))}
                          </div>
                        </div>
                        <span className="text-xs font-bold text-amber-400 shrink-0">★ {item.rating}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* ALPHA WEIGHT CONTROL */}
              <div className={`border rounded-xl p-5 flex flex-col md:flex-row items-center justify-between gap-4 ${bgCard}`}>
                <div className="flex items-center gap-3">
                  <div className="p-2.5 rounded-lg bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">
                    <Sliders className="h-5 w-5" />
                  </div>
                  <div>
                    <h3 className={`text-sm font-semibold ${isDark ? 'text-slate-100' : 'text-slate-900'}`}>Hybrid Fusion Weighting ($\alpha$)</h3>
                    <p className={`text-xs ${textMuted}`}>Blend Collaborative Filtering (ALS) with Content-Based (TF-IDF)</p>
                  </div>
                </div>

                <div className={`flex items-center gap-4 w-full md:w-96 p-3 rounded-xl border ${isDark ? 'bg-slate-950 border-slate-800' : 'bg-slate-100 border-slate-300'}`}>
                  <div className="text-right shrink-0">
                    <span className="text-xs font-bold text-cyan-400 block">CF {Math.round(alphaWeight * 100)}%</span>
                    <span className={`text-[10px] ${textMuted}`}>ALS Latent</span>
                  </div>

                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={alphaWeight}
                    onChange={(e) => setAlphaWeight(parseFloat(e.target.value))}
                    className="w-full accent-cyan-400 h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer"
                  />

                  <div className="text-left shrink-0">
                    <span className="text-xs font-bold text-purple-400 block">CB {Math.round((1 - alphaWeight) * 100)}%</span>
                    <span className={`text-[10px] ${textMuted}`}>TF-IDF Meta</span>
                  </div>
                </div>
              </div>

              {/* RECOMMENDATION RESULTS GRID */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <h3 className={`text-sm font-semibold flex items-center gap-2 ${isDark ? 'text-slate-200' : 'text-slate-800'}`}>
                    <Award className="h-4 w-4 text-cyan-400" /> Top-N Recommended Candidates (Real-time Fusion)
                  </h3>
                  <span className={`text-xs font-mono ${textMuted}`}>Excluding Seen Train Items</span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {hybridRecommendations.map((item, rank) => (
                    <div 
                      key={item.id} 
                      className={`border rounded-xl p-4 transition-all duration-200 flex flex-col justify-between space-y-4 group relative overflow-hidden ${bgCard}`}
                    >
                      <div className="absolute top-0 right-0 bg-cyan-500/20 text-cyan-300 text-[10px] font-mono font-bold px-2.5 py-1 rounded-bl-lg border-b border-l border-cyan-500/30">
                        RANK #{rank + 1}
                      </div>

                      <div className="space-y-2">
                        <div className="flex items-center gap-2">
                          <span className="text-xs font-mono text-cyan-400 bg-cyan-950/60 border border-cyan-800/60 px-2 py-0.5 rounded">
                            {item.id}
                          </span>
                          <span className={`text-xs ${textMuted}`}>{item.category}</span>
                        </div>
                        <h4 className={`text-sm font-semibold transition-colors ${isDark ? 'text-slate-100 group-hover:text-cyan-300' : 'text-slate-900 group-hover:text-cyan-600'}`}>
                          {item.name}
                        </h4>
                        <p className={`text-xs line-clamp-2 leading-relaxed ${textMuted}`}>
                          {item.desc}
                        </p>
                      </div>

                      <div className={`space-y-3 pt-2 border-t ${borderCol}`}>
                        <div className="flex flex-wrap gap-1">
                          {item.tags.map((t, i) => (
                            <span key={i} className={`text-[10px] px-2 py-0.5 rounded border ${isDark ? 'bg-slate-950 border-slate-800 text-slate-400' : 'bg-slate-100 border-slate-300 text-slate-600'}`}>
                              #{t}
                            </span>
                          ))}
                        </div>

                        <div className={`p-2.5 rounded-lg border space-y-1.5 ${isDark ? 'bg-slate-950 border-slate-800' : 'bg-slate-100 border-slate-200'}`}>
                          <div className="flex justify-between items-center text-xs">
                            <span className={`font-medium ${textMuted}`}>Ensemble Score:</span>
                            <span className="font-bold font-mono text-emerald-400 text-sm">
                              {item.hybridScore.toFixed(3)}
                            </span>
                          </div>

                          <div className="w-full bg-slate-800 h-1.5 rounded-full flex overflow-hidden">
                            <div className="bg-cyan-500 h-full" style={{ width: `${item.cfScore * alphaWeight * 100}%` }} />
                            <div className="bg-purple-500 h-full" style={{ width: `${item.cbScore * (1 - alphaWeight) * 100}%` }} />
                          </div>

                          <div className={`flex justify-between text-[10px] font-mono pt-1 ${textMuted}`}>
                            <span>CF: <strong className="text-cyan-400">{item.cfScore.toFixed(3)}</strong></span>
                            <span>CB: <strong className="text-purple-400">{item.cbScore.toFixed(3)}</strong></span>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* TAB 2: MODEL BENCHMARK LEADERBOARD */}
          {activeTab === 'benchmark' && (
            <div className="space-y-6">
              
              {/* FILTER & SPLIT BAR */}
              <div className={`border p-4 rounded-xl flex flex-col sm:flex-row items-center justify-between gap-4 ${bgCard}`}>
                <div className="flex items-center gap-3 w-full sm:w-auto">
                  <Search className="h-4 w-4 text-cyan-400" />
                  <input
                    type="text"
                    placeholder="Search model name (e.g., ALS, BPR)..."
                    value={modelSearchQuery}
                    onChange={(e) => setModelSearchQuery(e.target.value)}
                    className={`text-xs rounded-lg px-3 py-2 outline-none border w-full sm:w-64 ${inputBg}`}
                  />
                </div>

                <div className="flex bg-slate-900 p-1 rounded-lg border border-slate-800">
                  <button
                    onClick={() => setBenchmarkSplit('Validation')}
                    className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                      benchmarkSplit === 'Validation' ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/40' : 'text-slate-400'
                    }`}
                  >
                    Validation Split
                  </button>
                  <button
                    onClick={() => setBenchmarkSplit('Test')}
                    className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                      benchmarkSplit === 'Test' ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/40' : 'text-slate-400'
                    }`}
                  >
                    Test Holdout Split
                  </button>
                </div>
              </div>

              {/* LEADERBOARD TABLE */}
              <div className={`border rounded-xl overflow-hidden ${bgCard}`}>
                <div className={`px-5 py-4 border-b flex items-center justify-between ${borderCol}`}>
                  <h3 className={`text-sm font-semibold ${isDark ? 'text-slate-200' : 'text-slate-800'}`}>Full-Catalog Ranking Leaderboard ({benchmarkSplit})</h3>
                  <span className={`text-xs font-mono ${textMuted}`}>Click column headers to sort</span>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead className={`font-mono text-[11px] uppercase tracking-wider border-b ${borderCol} ${isDark ? 'bg-slate-950 text-slate-400' : 'bg-slate-100 text-slate-600'}`}>
                      <tr>
                        <th className="py-3 px-4 cursor-pointer hover:text-cyan-400" onClick={() => handleSort('model')}>
                          <span className="flex items-center gap-1">Model <ArrowUpDown className="h-3 w-3"/></span>
                        </th>
                        <th className="py-3 px-4 cursor-pointer hover:text-cyan-400" onClick={() => handleSort('hr5')}>
                          <span className="flex items-center gap-1">HitRate@5 <ArrowUpDown className="h-3 w-3"/></span>
                        </th>
                        <th className="py-3 px-4 cursor-pointer hover:text-cyan-400" onClick={() => handleSort('hr10')}>
                          <span className="flex items-center gap-1">HitRate@10 <ArrowUpDown className="h-3 w-3"/></span>
                        </th>
                        <th className="py-3 px-4 cursor-pointer hover:text-cyan-400" onClick={() => handleSort('hr20')}>
                          <span className="flex items-center gap-1">HitRate@20 <ArrowUpDown className="h-3 w-3"/></span>
                        </th>
                        <th className="py-3 px-4 cursor-pointer hover:text-cyan-400" onClick={() => handleSort('ndcg10')}>
                          <span className="flex items-center gap-1">NDCG@10 <ArrowUpDown className="h-3 w-3"/></span>
                        </th>
                        <th className="py-3 px-4 cursor-pointer hover:text-cyan-400" onClick={() => handleSort('mrr')}>
                          <span className="flex items-center gap-1">MRR <ArrowUpDown className="h-3 w-3"/></span>
                        </th>
                        <th className="py-3 px-4 text-right cursor-pointer hover:text-cyan-400" onClick={() => handleSort('timeSec')}>
                          <span className="flex items-center justify-end gap-1">Fit Time <ArrowUpDown className="h-3 w-3"/></span>
                        </th>
                      </tr>
                    </thead>
                    <tbody className={`divide-y font-mono ${isDark ? 'divide-slate-800/60' : 'divide-slate-200'}`}>
                      {processedBenchmarkData.map((row, idx) => (
                        <tr key={idx} className={`transition-colors ${isDark ? 'hover:bg-slate-800/40' : 'hover:bg-slate-50'}`}>
                          <td className="py-3 px-4 font-bold text-cyan-400">{row.model}</td>
                          <td className="py-3 px-4">{row.hr5.toFixed(3)}</td>
                          <td className="py-3 px-4 font-bold text-indigo-400">{row.hr10.toFixed(3)}</td>
                          <td className="py-3 px-4">{row.hr20.toFixed(3)}</td>
                          <td className="py-3 px-4 text-emerald-400">{row.ndcg10.toFixed(3)}</td>
                          <td className="py-3 px-4 text-pink-400">{row.mrr.toFixed(3)}</td>
                          <td className="py-3 px-4 text-right text-slate-400">{row.timeSec}s</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* CHARTS COMPARISON */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className={`border rounded-xl p-5 space-y-4 ${bgCard}`}>
                  <h3 className={`text-sm font-semibold ${isDark ? 'text-slate-200' : 'text-slate-800'}`}>HitRate@K Comparison</h3>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={processedBenchmarkData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke={isDark ? '#1e293b' : '#cbd5e1'} />
                        <XAxis dataKey="model" stroke="#64748b" fontSize={12} />
                        <YAxis stroke="#64748b" fontSize={12} domain={[0, 0.6]} />
                        <Tooltip contentStyle={{ backgroundColor: isDark ? '#0f172a' : '#ffffff', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }} />
                        <Legend wrapperStyle={{ fontSize: '12px' }} />
                        <Bar dataKey="hr5" name="HitRate@5" fill="#38bdf8" radius={[4, 4, 0, 0]} />
                        <Bar dataKey="hr10" name="HitRate@10" fill="#818cf8" radius={[4, 4, 0, 0]} />
                        <Bar dataKey="hr20" name="HitRate@20" fill="#c084fc" radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className={`border rounded-xl p-5 space-y-4 ${bgCard}`}>
                  <h3 className={`text-sm font-semibold ${isDark ? 'text-slate-200' : 'text-slate-800'}`}>NDCG@10 vs MRR Quality</h3>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={processedBenchmarkData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke={isDark ? '#1e293b' : '#cbd5e1'} />
                        <XAxis dataKey="model" stroke="#64748b" fontSize={12} />
                        <YAxis stroke="#64748b" fontSize={12} domain={[0, 0.35]} />
                        <Tooltip contentStyle={{ backgroundColor: isDark ? '#0f172a' : '#ffffff', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }} />
                        <Legend wrapperStyle={{ fontSize: '12px' }} />
                        <Line type="monotone" dataKey="ndcg10" name="NDCG@10" stroke="#34d399" strokeWidth={2.5} />
                        <Line type="monotone" dataKey="mrr" name="MRR" stroke="#f472b6" strokeWidth={2.5} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* TAB 3: SPARK & ARROW STAGING INSPECTOR */}
          {activeTab === 'staging' && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className={`border p-4 rounded-xl space-y-1 ${bgCard}`}>
                  <div className={`flex items-center justify-between text-xs ${textMuted}`}>
                    <span>Sparse CSR Matrix</span>
                    <Database className="h-3.5 w-3.5 text-cyan-400" />
                  </div>
                  <div className="text-xl font-bold font-mono">142.8K × 68.4K</div>
                  <div className={`text-[11px] font-mono ${textMuted}`}>Users × Items Shape</div>
                </div>

                <div className={`border p-4 rounded-xl space-y-1 ${bgCard}`}>
                  <div className={`flex items-center justify-between text-xs ${textMuted}`}>
                    <span>Positive NNZ Count</span>
                    <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />
                  </div>
                  <div className="text-xl font-bold text-emerald-400 font-mono">1,284,910</div>
                  <div className={`text-[11px] font-mono ${textMuted}`}>Ratings ≥ 4.0 Binarized</div>
                </div>

                <div className={`border p-4 rounded-xl space-y-1 ${bgCard}`}>
                  <div className={`flex items-center justify-between text-xs ${textMuted}`}>
                    <span>Disk Staging Engine</span>
                    <HardDrive className="h-3.5 w-3.5 text-indigo-400" />
                  </div>
                  <div className="text-xl font-bold text-indigo-400 font-mono">Tempfile Parquet</div>
                  <div className={`text-[11px] font-mono ${textMuted}`}>Zero Java Heap OOM</div>
                </div>

                <div className={`border p-4 rounded-xl space-y-1 ${bgCard}`}>
                  <div className={`flex items-center justify-between text-xs ${textMuted}`}>
                    <span>Driver Memory Limit</span>
                    <CpuIcon className="h-3.5 w-3.5 text-purple-400" />
                  </div>
                  <div className="text-xl font-bold text-purple-400 font-mono">12.0 GB RAM</div>
                  <div className={`text-[11px] font-mono ${textMuted}`}>PyArrow Batch Stream</div>
                </div>
              </div>

              <div className={`border rounded-xl p-5 space-y-4 ${bgCard}`}>
                <h3 className={`text-sm font-semibold flex items-center gap-2 ${isDark ? 'text-slate-200' : 'text-slate-800'}`}>
                  <Terminal className="h-4 w-4 text-cyan-400" /> PySpark Memory Safety & Parquet Staging Architecture
                </h3>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                  <div className={`p-4 rounded-lg border space-y-2 ${isDark ? 'bg-slate-950 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
                    <div className="font-bold text-cyan-400 flex items-center gap-2">
                      <ShieldCheck className="h-3.5 w-3.5" /> Temporary Directory Staging
                    </div>
                    <p className={`leading-relaxed text-[11px] ${textMuted}`}>
                      By bypassing direct <code className="font-mono text-cyan-400">.collect()</code> calls on millions of ground-truth rows, Spark writes partitioned Parquet files into <code className="font-mono text-cyan-400">tempfile.TemporaryDirectory()</code>. PyArrow reads these batches efficiently without triggering Java heap garbage collection crashes.
                    </p>
                  </div>

                  <div className={`p-4 rounded-lg border space-y-2 ${isDark ? 'bg-slate-950 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
                    <div className="font-bold text-emerald-400 flex items-center gap-2">
                      <ShieldCheck className="h-3.5 w-3.5" /> Warm-Start Filtering
                    </div>
                    <p className={`leading-relaxed text-[11px] ${textMuted}`}>
                      Evaluation protocols strictly verify user existence across both train and validation splits. Cold-start items or inactive users are automatically filtered using Spark Left Anti-Joins.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* TAB 4: A/B EVAL SUBSAMPLE SIMULATOR */}
          {activeTab === 'simulator' && (
            <div className="space-y-6 max-w-4xl">
              <div className={`border rounded-xl p-6 space-y-6 ${bgCard}`}>
                <div>
                  <h3 className={`text-base font-semibold mb-1 ${isDark ? 'text-slate-100' : 'text-slate-900'}`}>A/B Evaluation Subsample Simulator</h3>
                  <p className={`text-xs ${textMuted}`}>Test recommendation benchmarks rapidly by randomly shuffling and sampling evaluation user subsets.</p>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                  <div className="space-y-2">
                    <label className={`text-xs font-semibold flex justify-between ${isDark ? 'text-slate-300' : 'text-slate-700'}`}>
                      <span>Evaluation Subsample Ratio</span>
                      <span className="text-cyan-400 font-mono">{Math.round(evalRatio * 100)}% Users</span>
                    </label>
                    <input
                      type="range"
                      min="0.1"
                      max="1.0"
                      step="0.1"
                      value={evalRatio}
                      onChange={(e) => setEvalRatio(parseFloat(e.target.value))}
                      className="w-full accent-cyan-400 h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer"
                    />
                    <p className={`text-[11px] ${textMuted}`}>Set to 20% or 50% for instant notebook experimentation.</p>
                  </div>

                  <div className="space-y-2">
                    <label className={`text-xs font-semibold flex justify-between ${isDark ? 'text-slate-300' : 'text-slate-700'}`}>
                      <span>Negative Items Sampled per User</span>
                      <span className="text-emerald-400 font-mono">{negativesPerUser} Items</span>
                    </label>
                    <input
                      type="range"
                      min="19"
                      max="99"
                      step="20"
                      value={negativesPerUser}
                      onChange={(e) => setNegativesPerUser(parseInt(e.target.value))}
                      className="w-full accent-emerald-400 h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer"
                    />
                    <p className={`text-[11px] ${textMuted}`}>Full catalog negative sampling for ranking evaluation.</p>
                  </div>
                </div>

                <div className={`p-4 rounded-xl border flex items-center justify-between ${isDark ? 'bg-slate-950 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
                  <div>
                    <h4 className={`text-xs font-semibold ${isDark ? 'text-slate-200' : 'text-slate-800'}`}>Estimated Evaluation Execution Time</h4>
                    <p className={`text-[11px] ${textMuted}`}>~{Math.round(evalRatio * 48)} seconds on Kaggle CPU (12GB RAM)</p>
                  </div>
                  <button
                    onClick={runSimulation}
                    className="bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold px-4 py-2 rounded-lg text-xs transition-all shadow-md shadow-cyan-500/20"
                  >
                    Run Subsample Eval
                  </button>
                </div>
              </div>
            </div>
          )}

        </div>
      </main>

    </div>
  );
}