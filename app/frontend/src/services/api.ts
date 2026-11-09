// API service layer
const BASE = '';

export async function fetchSampleUsers(n = 12) {
  const res = await fetch(`/users/sample?n=${n}`);
  return res.json();
}

export async function fetchModels() {
  const res = await fetch('/models');
  return res.json();
}

export async function fetchRecommendations(params: {
  customer_id?: string;
  user_idx?: number;
  model: string;
  k: number;
  alpha: number;
  history?: number[];
}) {
  const res = await fetch('/recommend', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function fetchMetrics() {
  const res = await fetch('/metrics');
  return res.json();
}

export async function runSimulation(params: { users: number; negatives: number; alpha: number }) {
  const res = await fetch('/simulate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
