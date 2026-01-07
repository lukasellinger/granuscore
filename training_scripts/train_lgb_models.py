from typing import List, Tuple, Dict

import numpy as np
import lightgbm as lgb
import optuna

from training_scripts.config import PROJECT_DIR
from training_scripts.utils import stack_answer_features, load_sim_data, pairwise_order_accuracy


def count_consecutive_ids(items, key="id"):
    counts = []
    prev = None
    current_count = 0

    for item in items:
        curr = item[key]
        if curr != prev:
            if prev is not None:
                counts.append(current_count)
            current_count = 1
            prev = curr
        else:
            current_count += 1

    if prev is not None:
        counts.append(current_count)

    return counts


def build_lgb_dataset(sim_data: List[Dict], model_type, k=1000) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    X = np.stack([stack_answer_features(e, model_type, k) for e in sim_data])
    y = np.asarray([e["Y"] for e in sim_data])
    groups = count_consecutive_ids(sim_data)
    return X, y, groups


def evaluate_oder_acc(model, X_val, y_val):
    preds = model.predict(X_val, num_iteration=model.best_iteration)
    acc = pairwise_order_accuracy(preds, y_val)
    return float(acc) if not np.isnan(acc) else -1.0


def mean_group_order_acc(y_true, preds, groups):
    accs = []
    idx = 0

    for g in groups:
        y_g = y_true[idx:idx + g]
        p_g = preds[idx:idx + g]

        if len(np.unique(y_g)) > 1:
            acc = pairwise_order_accuracy(p_g, y_g)
            if not np.isnan(acc):
                accs.append(acc)

        idx += g

    if not accs:
        return -1.0

    return float(np.mean(accs))


def optuna_regression(
    X_train,
    y_train,
    groups_train,
    X_val,
    y_val,
    groups_val,
    n_trials: int = 50,
):
    def objective(trial):
        params = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        }

        train_ds = lgb.Dataset(X_train, label=y_train)
        val_ds = lgb.Dataset(X_val, label=y_val)

        model = lgb.train(
            params,
            train_ds,
            valid_sets=[val_ds],
            num_boost_round=10000,
            callbacks=[lgb.early_stopping(200, verbose=False)],
        )

        preds = model.predict(X_val, num_iteration=model.best_iteration)

        acc_global = pairwise_order_accuracy(preds, y_val)
        acc_intra = mean_group_order_acc(y_val, preds, groups_val)

        alpha = 1.0  # tune; 0.5 = equal weight, 0.7 = favor global
        score = alpha * acc_global + (1 - alpha) * acc_intra

        return -1.0 if np.isnan(score) else score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def optuna_lambdarank(
    X_train,
    y_train,
    groups_train,
    X_val,
    y_val,
    groups_val,
    n_trials: int = 50,
):

    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        group=groups_train,
        params={"feature_pre_filter": False},
    )

    valid_data = lgb.Dataset(
        X_val,
        label=y_val,
        group=groups_val,
        params={"feature_pre_filter": False},

    )

    def objective(trial):
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [10],
            "label_gain": [1, 2, 3, 4],
            "verbosity": -1,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        }

        model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=5000,
            callbacks=[lgb.early_stopping(200, verbose=False)],
        )

        preds = model.predict(X_val, num_iteration=model.best_iteration)

        acc_global = pairwise_order_accuracy(preds, y_val)
        acc_intra = mean_group_order_acc(y_val, preds, groups_val)

        alpha = 0.7  # tune; 0.5 = equal weight, 0.7 = favor global
        score = alpha * acc_global + (1 - alpha) * acc_intra

        return -1.0 if np.isnan(score) else score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    return study.best_params

def make_groupwise_integer_labels(y, groups):
    """
    Convert (possibly normalized) float labels into
    integer relevance grades per group.

    Highest integer = most important (for LambdaRank).
    """
    y = np.asarray(y)
    new_y = np.zeros_like(y, dtype=np.int32)

    idx = 0
    for g in groups:
        y_g = y[idx:idx+g]

        # sort ascending (smallest = most important)
        order = np.argsort(y_g)

        # assign descending ranks: g-1 ... 0
        ranks = np.empty_like(order)
        ranks[order] = np.arange(g-1, -1, -1)

        new_y[idx:idx+g] = ranks
        idx += g

    return new_y


def train_single_model(optuna_type, model_type, prefix: str, k=1000, n_trials: int = 50):
    print(f"\n🚀 Training model for prefix='{prefix or 'full'}'")

    sim_data = {
        "train": load_sim_data(prefix, "train"),
        "val": load_sim_data(prefix, "eval"),
        "test": load_sim_data(prefix, "test"),
    }

    X_train, y_train, train_groups = build_lgb_dataset(sim_data["train"], model_type, k)
    X_val, y_val, val_groups = build_lgb_dataset(sim_data["val"], model_type, k)

    if optuna_type == 'regression':
        study_best_params = optuna_regression(X_train, y_train, train_groups, X_val, y_val, val_groups, n_trials)
    elif optuna_type == 'lambdarank':
        y_train = make_groupwise_integer_labels(y_train, train_groups)
        y_val = make_groupwise_integer_labels(y_val, val_groups)
        study_best_params = optuna_lambdarank(X_train, y_train, train_groups, X_val, y_val, val_groups, n_trials)
    else:
        raise ValueError(f"{optuna_type} is not a valid option")

    if optuna_type == 'regression':
        best_params = {
            **study_best_params,
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
        }

        final_model = lgb.train(
            best_params,
            lgb.Dataset(X_train, label=y_train),
            valid_sets=[lgb.Dataset(X_train, label=y_train), lgb.Dataset(X_val, label=y_val)],
            num_boost_round=10_000,
            callbacks=[
                lgb.early_stopping(200),
                lgb.log_evaluation(100),
            ],
        )
        val_acc = evaluate_oder_acc(final_model, X_val, y_val)
        out_path = f"{PROJECT_DIR}/src/granuscore/assets/lgb_models/{prefix}_model-new-k-{k}.txt"
    else:
        best_params = {
            **study_best_params,
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [10],
            "label_gain": [0, 1, 2, 3],
            "verbosity": -1,
        }

        train_dataset = lgb.Dataset(
                X_train,
                label=y_train,
                group=train_groups,
                params={"feature_pre_filter": False},
        )

        val_dataset = lgb.Dataset(
                X_val,
                label=y_val,
                group=val_groups,
                params={"feature_pre_filter": False},
        )

        final_model = lgb.train(
            best_params,
            train_dataset,
            valid_sets=[train_dataset, val_dataset],
            num_boost_round=10_000,
            callbacks=[
                lgb.early_stopping(200),
                lgb.log_evaluation(100),
            ],
        )
        val_acc = evaluate_oder_acc(final_model, X_val, y_val)
        out_path = f"{PROJECT_DIR}/src/granuscore/assets/lgb_models/{prefix}_model_lambdarank-new-k-{k}.txt"

    final_model.save_model(out_path, num_iteration=final_model.best_iteration)
    print(f"✅ Saved model → {out_path}")
    print(f"📊 Validation Acc: {val_acc:.4f}")

    return {
        "prefix": prefix or "full",
        "val_acc": val_acc,
        "best_iteration": final_model.best_iteration,
    }


def main():
    ablated_ks = [
        33,
        66,
        99,
        333,
        666,
        999,  # base
        1332,
        1665,
    ]
    ablated_ks = [999]

    results = []
    for k in ablated_ks:
        MODELS = [
            #(f"50k-hit-radial_anchors-{k}", "hit"),
            #(f"50k-hit-random_anchors-{k}", "hit"),
            #(f"50k-hit-random-{k}", "hit"),
            #(f"50k-hit-nearest_neighbor-{k}", "hit"),
            #(f"50k-all-min-random_anchors-{k}", "sentence_transformer"),
            #(f"50k-all-min-random-{k}", "sentence_transformer"),
            #(f"50k-all-min-nearest_neighbor-{k}", "sentence_transformer"),
            (f"50k-hit-embedding-radial_anchors", "hit-embedding"),
            (f"50k-hit-embedding-random_anchors", "hit-embedding"),
            (f"50k-hit-embedding-random", "hit-embedding"),
            (f"50k-hit-embedding-nearest_neighbor", "hit-embedding"),
        ]

        optuna_types = ['regression']
        for optuna_type in optuna_types:
            for prefix, model_type in MODELS:
                result = train_single_model(optuna_type, model_type, prefix, k, n_trials=50)
                results.append(result)

    # ---- ranking ----
    results = sorted(results, key=lambda x: x["val_acc"], reverse=True)

    print("\n🏆 MODEL RANKING (by pairwise acc)")
    print("-" * 50)
    for rank, r in enumerate(results, 1):
        print(
            f"{rank:>2}. {r['prefix']:<10} | "
            f"acc = {r['val_acc']:.4f} | "
            f"iter = {r['best_iteration']}"
        )

if __name__ == "__main__":
    main()
