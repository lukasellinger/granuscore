import pickle
from collections import defaultdict

import numpy as np
from scipy.stats import kendalltau, wilcoxon, pearsonr

from training_scripts.config import PROJECT_DIR


def stack_answer_features(e: dict, model_type, k=1000) -> np.ndarray:
    if model_type == "hit":
        return np.hstack([
            e["answer_entry_dist0"],
            e["answer_index_scores"][:k],
            e["answer_neighbors_dist"][:k],
        ])
    elif model_type == "hit-embedding":
        return np.hstack([
            e["answer_index_scores"][:k],
            e["answer_neighbors_dist"][:k],
        ])
    else:
        return e["answer_index_scores"]


def load_sim_data(prefix: str, split: str):
    name = f"{prefix}-{split}_data_lgb.pkl"
    path = f"{PROJECT_DIR}/data/precalc_data/{name}"
    with open(path, "rb") as f:
        return pickle.load(f)


def build_tau_dataset(
    split: str,
    hf_data,
    sim_data: dict[str, list[dict]],
    model_type: str,
    max_granola: int = 4,
    k: int = 1000,
) -> list[dict]:

    lookup = {
        f"{d['id']}-{d['Y']}": d
        for d in sim_data[split]
    }

    dataset = []

    for e in hf_data[split].to_list():
        answers, similarities, granularities = [], [], []

        for g in range(1, max_granola + 1):
            answer = e.get(f"granola_answer_{g}")
            if not answer:
                continue

            answers.append(answer)
            granularities.append(g)

        granularities = normalize_to_likert(granularities, max_granola)

        for g in granularities:
            sim_entry = lookup.get(f"{e['id']}-{g}")
            if sim_entry is None:
                continue

            similarities.append(stack_answer_features(sim_entry, model_type, k))

        dataset.append({
            "id": e["id"],
            "question": e["question"],
            "answers": answers,
            "similarities": similarities,
            "granularities": granularities,
        })

    return dataset


def run_evaluation(samples, predictor):
    return evaluate_preds(
        samples,
        pred_fn=lambda t: predictor.predict(t["similarities"]),
    )


def run_evaluation_dist0(samples):
    return evaluate_preds(
        samples,
        pred_fn=lambda t: [-float(sim[0]) for sim in t["similarities"]],
    )


def run_inverted_evaluation(samples):
    return evaluate_preds(
        samples,
        pred_fn=lambda t: [-float(sim[0]) for sim in t["similarities"]],
    )


def run_evaluation_llm(samples):
    return evaluate_preds(
        samples,
        pred_fn=lambda t: [float(sim[0]) for sim in t["similarities"]],
    )


def evaluate_preds(samples, pred_fn):
    pairwise_accs, taus, raw_taus, raw_pairwise_accs = [], [], [], []
    all_grouped_pairs = defaultdict(list)

    granu_by_position = defaultdict(list)
    full_correct = 0

    all_preds, all_ideals = [], []

    for test in samples:
        preds = pred_fn(test)
        ideal = test["granularities"]

        all_preds.extend(preds)
        all_ideals.extend([float(i) for i in ideal])

        grouped = get_value_groups(preds, ideal)
        for k, v in grouped.items():
            all_grouped_pairs[k].extend(v)

        acc = pairwise_order_accuracy(preds, ideal)
        tau, _ = kendalltau(ideal, preds)

        if not np.isnan(acc):
            pairwise_accs.append(acc)
        raw_pairwise_accs.append(acc)
        if not np.isnan(tau):
            taus.append(tau)
        raw_taus.append(tau)

        if acc == 1:
            full_correct += 1

        for g, p in zip(ideal, preds):
            granu_by_position[g].append(p)

    all_ideals_np = np.asarray(all_ideals)
    all_preds_np = np.asarray(all_preds)
    nan_mask = ~np.isnan(all_preds)

    global_pairwise_acc = pairwise_order_accuracy(all_preds_np[nan_mask], all_ideals_np[nan_mask])
    global_tau, _ = kendalltau(all_ideals_np[nan_mask], all_preds_np[nan_mask])
    global_pearson, _ = pearsonr(all_ideals_np[nan_mask], all_preds_np[nan_mask])

    stats = {
        "not_nan_percentage": np.mean(nan_mask) * 100,
        "global_pairwise_acc": r(global_pairwise_acc, 4),
        "global_kendall_tau": r(global_tau, 4),
        "global_pearson": r(global_pearson, 4),
        "avg_intra_pairwise_acc": r(np.mean(pairwise_accs), 4),
        "avg_intra_kendall_tau": r(np.mean(taus), 4),
        "mean_intra_full_correct": r(full_correct / len(pairwise_accs), 4),
        "avg_granuscores": {
            float(g): r(np.mean(granu_by_position[g]), 4)
            for g in sorted(granu_by_position)
        },
        "intra_significance": compute_significance(all_grouped_pairs),
        "intra_kendall_taus": raw_taus,
        "intra_pairwise_acc": raw_pairwise_accs,
        "predicted_grans": all_preds,
        "gold_grans": all_ideals,
    }

    return stats


def compute_significance(grouped_pairs):
    results = []

    for (i, j), values in sorted(grouped_pairs.items()):
        if not values:
            continue

        a = np.array([v[0] for v in values])
        b = np.array([v[1] for v in values])

        mean_a = np.mean(a)
        mean_b = np.mean(b)

        direction = "i<j ✔" if mean_a < mean_b else "i>j ✖"

        _, pval = wilcoxon(a, b, alternative="less")

        results.append(
            f"Comp: {i} vs {j}: "
            f"mean_i={mean_a:.4f}, mean_j={mean_b:.4f}, "
            f"p={pval:.3e}, n={len(a)}, "
            f"{'Significant' if pval < 0.01 else ''} "
            f"({direction})"
        )

    return results


def r(x, ndigits=2):
    return None if x is None else round(float(x), ndigits)


def bootstrap_paired_diff(
    A,
    B,
    n_boot=20000,
    seed=0,
    alternative="greater",  # H1: A > B
    confidence=0.95,
):
    A = np.array(
        [np.nan if v is None else v for v in A],
        dtype=float
    )
    B = np.array(
        [np.nan if v is None else v for v in B],
        dtype=float
    )

    if len(A) != len(B):
        raise ValueError("A and B must have same length")

    # paired deletion
    mask = np.isfinite(A) & np.isfinite(B)
    A, B = A[mask], B[mask]
    n = len(A)

    rng = np.random.default_rng(seed)
    diffs = A - B
    obs = diffs.mean()

    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = diffs[idx].mean()

    alpha = 1 - confidence
    ci = (
        np.quantile(boot, alpha / 2),
        np.quantile(boot, 1 - alpha / 2),
    )

    if alternative == "greater":
        p = np.mean(boot <= 0)
    elif alternative == "less":
        p = np.mean(boot >= 0)
    elif alternative == "two-sided":
        p = 2 * min(np.mean(boot <= 0), np.mean(boot >= 0))
    else:
        raise ValueError("Invalid alternative")

    return {
        "n": n,
        "mean_diff": float(obs),
        "ci": tuple(map(float, ci)),
        "p_value": float(p),
        "boot_diffs": boot.tolist(),
    }


def bootstrap_pairwise_acc_diff(
    gold,
    preds_A,
    preds_B,
    n_boot=20000,
    seed=0,
    alternative="greater",  # "two-sided", "greater" (A>B), "less" (A<B)
    confidence=0.95,
):
    """
    Paired bootstrap test for difference in pairwise acc: acc(A) - acc(B).

    Returns:
      - acc_A, acc_B, delta
      - bootstrap deltas, CI
      - p_value for chosen alternative
    """
    gold = np.asarray(gold)
    preds_A = np.asarray(preds_A)
    preds_B = np.asarray(preds_B)

    if not (len(gold) == len(preds_A) == len(preds_B)):
        raise ValueError("gold, preds_A, preds_B must have the same length")

    # Drop rows with any non-finite values
    mask = np.isfinite(gold) & np.isfinite(preds_A) & np.isfinite(preds_B)
    gold = gold[mask]
    preds_A = preds_A[mask]
    preds_B = preds_B[mask]

    n = len(gold)
    if n < 3:
        raise ValueError("Need at least 3 valid samples to compute Kendall's tau")

    # Observed stats
    acc_A = pairwise_order_accuracy(preds_A, gold)
    acc_B = pairwise_order_accuracy(preds_B, gold)

    delta_obs = acc_A - acc_B

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=float)

    # Paired bootstrap: resample indices, evaluate both models on same resample
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)  # resample with replacement
        tA = pairwise_order_accuracy(preds_A[idx], gold[idx])
        tB = pairwise_order_accuracy(preds_B[idx], gold[idx])
        deltas[b] = tA - tB

    # CI
    alpha = 1.0 - confidence
    lo = np.quantile(deltas, alpha / 2)
    hi = np.quantile(deltas, 1 - alpha / 2)

    # p-value via bootstrap distribution around 0
    if alternative == "two-sided":
        p = 2 * min(np.mean(deltas <= 0), np.mean(deltas >= 0))
        p = min(p, 1.0)
    elif alternative == "greater":  # A > B
        p = np.mean(deltas <= 0)
    elif alternative == "less":     # A < B
        p = np.mean(deltas >= 0)
    else:
        raise ValueError("alternative must be one of: 'two-sided', 'greater', 'less'")

    return {
        "n_used": n,
        "acc_A": float(acc_A),
        "acc_B": float(acc_B),
        "delta_obs": float(delta_obs),
        "bootstrap_deltas": deltas.tolist(),
        "ci": (float(lo), float(hi)),
        "p_value": float(p),
        "alternative": alternative,
        "confidence": confidence,
        "n_boot": n_boot,
        "seed": seed,
    }


def get_value_groups(pred, ideal):
    """
    Return grouped cross-cluster predicted pairs:
    grouped["a-b"] = [(pred_i, pred_j), ...]
    where ideal_i = a and ideal_j = b and a != b.
    """

    pred = np.asarray(pred)
    ideal = np.asarray(ideal)

    n = len(pred)
    grouped = defaultdict(list)

    # All i < j pairs
    i, j = np.triu_indices(n, k=1)

    # Keep only comparable gold pairs
    mask = ideal[i] != ideal[j]
    i = i[mask]
    j = j[mask]

    # Vectorized extraction
    ideal_i = ideal[i]
    ideal_j = ideal[j]
    pred_i = pred[i]
    pred_j = pred[j]

    # Now grouping (small Python loop over filtered pairs)
    for a, b, pi, pj in zip(ideal_i, ideal_j, pred_i, pred_j):
        grouped[(a,b)].append((pi, pj))

    return grouped


def pairwise_order_score(pairs) -> float | None:
    """Scaled pairwise ordering accuracy on comparable pairs: 2*acc - 1 (ties count as wrong)."""
    if not pairs:
        return None

    pairs = np.asarray(pairs, dtype=np.float32)

    p_i = pairs[:, 0]
    p_j = pairs[:, 1]
    ideal_i = pairs[:, 2]
    ideal_j = pairs[:, 3]

    concordant = ((p_i < p_j) & (ideal_i < ideal_j)) | \
                 ((p_i > p_j) & (ideal_i > ideal_j))

    concordant_count = np.sum(concordant)
    total = len(pairs)

    discordant_count = total - concordant_count

    return (concordant_count - discordant_count) / total


def pairwise_order_accuracy(pred, ideal, answers=None, ignore_nans=True):
    """
    Compute pairwise ranking accuracy on comparable pairs.

    - Only considers pairs where ideal_i != ideal_j
    - Counts prediction ties as incorrect
    - Returns accuracy in [0, 1]
    """
    pred = np.asarray(pred)
    ideal = np.asarray(ideal)
    answers = np.asarray(answers) if answers is not None else None

    if ignore_nans:
        mask = np.isfinite(pred) & np.isfinite(ideal)
        pred = pred[mask]
        ideal = ideal[mask]
        answers = answers[mask] if answers is not None else None
    else:
        if np.isnan(pred).any() or np.isnan(ideal).any():
            return np.nan

    n = pred.shape[0]
    if n < 2:
        return np.nan

    # All i < j index pairs
    i, j = np.triu_indices(n, k=1)

    # Keep only comparable pairs (gold differs)
    mask = ideal[i] != ideal[j]
    if answers is not None:
        answers = np.char.strip(np.char.lower(answers))
        mask &= answers[i] != answers[j]

    if not np.any(mask):
        return np.nan

    i = i[mask]
    j = j[mask]

    # Correct if predicted and gold differences have same sign
    pred_diff = pred[i] - pred[j]
    ideal_diff = ideal[i] - ideal[j]

    correct = (pred_diff * ideal_diff) > 0

    return float(np.mean(correct))


def normalize_to_likert(seq, L=4):
    seq = np.asarray(seq)
    n = len(seq)

    if n == 1:
        return np.array([1.0])

    ranks = np.arange(n)
    scaled = 1 + ranks * (L - 1) / (n - 1)

    return scaled.astype(np.float32)

if __name__ == "__main__":
    print(pairwise_order_accuracy([1, 2], [1, 2]))
    print(pairwise_order_accuracy([1], [1]))

    print(normalize_to_likert([1, 2]))
    print(normalize_to_likert([1, 2, 3]))
    print(normalize_to_likert([1, 2, 3, 4]))