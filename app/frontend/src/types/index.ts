// Shared types for the storefront UI
export interface Product {
  item_idx: number;
  asin: string;
  title: string;
  brand: string;
  category: string;
  price: string;
  description: string;
  rating?: number;
  reviews_count?: number;
  tags?: string[];
  in_stock?: boolean;
}

export interface SampleUser {
  customer_id: string;
  user_idx: number;
  label: string;
  tier: string;
  interactions: number;
}

export interface ResolvedUser {
  customer_id: string;
  user_idx: number;
  user_status: string;
}

export interface ModelInfo {
  id: string;
  name: string;
  type: string;
  desc: string;
}

export interface BenchmarkRow {
  model: string;
  type: string;
  hr5: number;
  hr10: number;
  hr20: number;
  ndcg5: number;
  ndcg10: number;
  ndcg20: number;
  mrr: number;
  timeSec: number;
  status: string;
  best: boolean;
}

export interface SimResult {
  n_users: number;
  negatives: number;
  alpha: number;
  models: {
    id: string;
    name: string;
    hr5: number;
    hr10: number;
    hr20: number;
    ndcg10: number;
    mrr: number;
  }[];
}
