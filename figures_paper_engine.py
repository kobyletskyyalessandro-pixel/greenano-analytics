# figures_paper_engine.py
import numpy as np
import pandas as pd

# ----------------------------
# Core utilities (FiguresPaper-like)
# ----------------------------
def dirichlet_samples(alpha: np.ndarray, n: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    alpha = np.asarray(alpha, dtype=float)
    alpha = np.clip(alpha, 1e-9, None)
    return rng.dirichlet(alpha, size=int(n))

def weighted_geometric_mean(scores: np.ndarray, weights: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    scores: (N, K) positive
    weights: (K,) sums to 1
    returns: (N,)
    """
    s = np.asarray(scores, dtype=float)
    w = np.asarray(weights, dtype=float)
    w = w / max(w.sum(), eps)
    s = np.clip(s, eps, None)
    return np.exp(np.dot(np.log(s), w))

def spearman_rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    """
    Spearman via ranks (no scipy dependency).
    """
    a = np.asarray(a)
    b = np.asarray(b)
    ra = pd.Series(a).rank(method="average").to_numpy()
    rb = pd.Series(b).rank(method="average").to_numpy()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = (np.sqrt((ra**2).sum()) * np.sqrt((rb**2).sum()))
    if denom == 0:
        return 0.0
    return float((ra * rb).sum() / denom)

# ----------------------------
# Paper-style analyses
# ----------------------------
def preference_sweep_ops(
    df: pd.DataFrame,
    score_cols: list[str],              # e.g. ["P1_score","P2_score","P3_score"]
    base_weights: np.ndarray,           # e.g. [1/3,1/3,1/3]
    n_sims: int = 2000,
    concentration: float = 50.0,        # higher = weights closer to base_weights
    seed: int | None = 0,
) -> pd.DataFrame:
    """
    Simula incertezza sui pesi (Dirichlet attorno ai base_weights)
    e misura la stabilità del ranking su OPS.
    Ritorna una tabella con: top1_freq, mean_rank, std_rank, etc.
    """
    if len(score_cols) == 0:
        raise ValueError("score_cols vuoto")

    w0 = np.asarray(base_weights, dtype=float)
    w0 = w0 / max(w0.sum(), 1e-12)

    # alpha = concentration * w0 + 1
    alpha = concentration * w0 + 1.0
    W = dirichlet_samples(alpha, n_sims, seed=seed)  # (n_sims, K)

    S = df[score_cols].to_numpy(dtype=float)          # (N, K)
    S = np.clip(S, 1e-12, None)

    # OPS simulations: (n_sims, N)
    OPS_sim = np.exp(W @ np.log(S).T)

    # ranks: 1 = best
    ranks = np.argsort(np.argsort(-OPS_sim, axis=1), axis=1) + 1  # (n_sims,N)

    top1 = (ranks == 1).sum(axis=0) / n_sims
    mean_rank = ranks.mean(axis=0)
    std_rank = ranks.std(axis=0)

    out = pd.DataFrame({
        "Material_Name": df["Material_Name"].astype(str).to_numpy(),
        "top1_freq": top1,
        "mean_rank": mean_rank,
        "std_rank": std_rank,
    }).sort_values(["top1_freq", "mean_rank"], ascending=[False, True])

    return out

def threshold_sensitivity_simple(
    df: pd.DataFrame,
    raw_col: str,                 # e.g. "P1"
    thresholds_list: list[list[float]],
    score_levels: list[float],    # e.g. [0.2,0.4,0.6,0.8,1.0]
) -> pd.DataFrame:
    """
    Prova diverse scelte di soglie e misura quanto cambia il ranking (Spearman su score).
    Non “rifà tutta l’app”, è un test paper-friendly e leggero.
    """
    x = df[raw_col].to_numpy(dtype=float)

    def discretize(vals, thr, levels):
        thr = sorted(list(thr))
        out = np.empty_like(vals, dtype=float)
        for i, v in enumerate(vals):
            idx = 0
            while idx < len(thr) and v > thr[idx]:
                idx += 1
            idx = min(idx, len(levels) - 1)
            out[i] = levels[idx]
        return out

    base = discretize(x, thresholds_list[0], score_levels)
    rows = []
    for thr in thresholds_list:
        cur = discretize(x, thr, score_levels)
        rho = spearman_rank_corr(base, cur)
        rows.append({"raw_col": raw_col, "thresholds": str(thr), "spearman_vs_base": rho})
    return pd.DataFrame(rows).sort_values("spearman_vs_base", ascending=False)