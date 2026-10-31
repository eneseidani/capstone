"""S05: fixed-benchmark, fixed-metric, pooled cross-asset classification.

One self-contained Python file. Train on mature 60-session labels only.
Model selection uses pre-2023 purged chronological validation; 2023+ is report-only.
Metrics/threshold/baseline are FROZEN across model families.
"""
from __future__ import annotations
import argparse
import json
import time
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, precision_score, recall_score,
                             accuracy_score, balanced_accuracy_score, matthews_corrcoef,
                             average_precision_score, f1_score, log_loss, confusion_matrix)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterSampler
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LOG = logging.getLogger("S05_classification")
BOUNDARY = pd.Timestamp("2023-01-01")
HORIZON = 60
THRESHOLD = 0.5
# ALL model families evaluated under the same rules. No threshold tuning on test.
MODEL_NAMES = ("HistoricalFrequency", "LogisticRegression", "RandomForest",
               "HistGradientBoosting", "XGBoost", "CatBoost")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, default=Path("features_data/combined_features.csv"))
    p.add_argument("--universe", type=Path, default=Path("Investable_Universe.csv"))
    p.add_argument("--label-lookup", type=Path, default=Path("model_output/S05_timing_audit/exact_label_end_lookup_pre2023.csv"))
    p.add_argument("--regimes", type=Path, default=Path("features_data/market_regimes.csv"))
    p.add_argument("--with-regime", action="store_true", help="Use regime features only when confirmed point-in-time available")
    p.add_argument("--out", type=Path, default=Path("model_output/S05_classification_tuned"))
    p.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-start", default="2000-01-01")
    p.add_argument("--trials", type=int, default=15,
                   help="Number of hyperparameter configurations per ML model, including the initial reference configuration")
    p.add_argument("--min-train-years", type=int, default=5,
                   help="Minimum complete calendar years of historical data before first annual tuning fold; default 5")
    return p.parse_args(argv)


def score(y, probability, threshold=THRESHOLD):
    """Immutable positive-class metric conventions, including zero-division behavior."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(probability, dtype=float)
    if not len(y) or not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid labels or predicted probabilities")
    yhat = (p >= threshold).astype(int)
    pclip = np.clip(p, 1e-15, 1-1e-15)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    has_both_classes = len(np.unique(y)) == 2
    return {"N": len(y), "Positive_Rate": float(y.mean()),
            "MCC": float(matthews_corrcoef(y, yhat)),
            "Balanced_Accuracy": float(balanced_accuracy_score(y, yhat)) if has_both_classes else np.nan,
            "PR_AUC": float(average_precision_score(y, p)) if has_both_classes else np.nan,
            "F1": float(f1_score(y, yhat, zero_division=0)),
            "Precision": float(precision_score(y, yhat, zero_division=0)),
            "ROC_AUC": float(roc_auc_score(y, p)) if has_both_classes else np.nan,
            "Log_Loss": float(log_loss(y, pclip, labels=[0, 1])),
            "Accuracy": float(accuracy_score(y, yhat)),
            "Recall": float(recall_score(y, yhat, zero_division=0)),
            "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)}

def load_data(a):
    uni = pd.read_csv(a.universe)
    must = {"Ticker", "Asset Class", "Source"}
    if not must <= set(uni):
        raise ValueError(f"Universe must contain {sorted(must)}")
    uni = uni.loc[uni.Source.astype(str).str.strip().str.casefold().eq("yahoo finance"),
                  ["Ticker", "Asset Class"]].copy()
    uni["Ticker"] = uni.Ticker.astype(str).str.strip()
    if uni.Ticker.duplicated().any() or uni["Asset Class"].isna().any():
        raise ValueError("Duplicate tickers or missing asset classes in universe")
    LOG.info("Reading %s", a.features)
    df = pd.read_csv(a.features, low_memory=False)
    required = {"Date", "Ticker", "Target_60d", "Forward_Return_60d"}
    if not required <= set(df) or df.columns.duplicated().any():
        raise ValueError("Missing essential or duplicated feature columns")
    df["Date"] = pd.to_datetime(df.Date, errors="raise").dt.normalize()
    df["Ticker"] = df.Ticker.astype(str)
    df = df.loc[df.Date.ge(pd.Timestamp(a.train_start))].copy()
    df = df.merge(uni, on="Ticker", how="inner", validate="many_to_one")
    if df.empty or df.duplicated(["Ticker", "Date"]).any():
        raise ValueError("No eligible rows or duplicated ticker-date keys")
    LOG.info("Eligible classes: %s", df.groupby("Asset Class").Ticker.nunique().to_dict())

    if a.with_regime:
        r = pd.read_csv(a.regimes, usecols=["Date", "Market_Regime"])
        r["Date"] = pd.to_datetime(r.Date, errors="raise").dt.normalize()
        r["Market_Regime"] = r.Market_Regime.astype(str).str.strip()
        if r.Date.duplicated().any() or not r.Market_Regime.isin(["Risk-On", "Risk-Off", "Transition"]).all():
            raise ValueError("Invalid regime file")
        if "Market_Regime" in df:
            raise ValueError("Ambiguous Market_Regime: already in features")
        date_map = pd.DataFrame({"Date": np.sort(df.Date.unique())})
        date_map = pd.merge_asof(date_map,
                      r.rename(columns={"Date": "Regime_Source_Date"}).sort_values("Regime_Source_Date"),
                      left_on="Date", right_on="Regime_Source_Date", direction="backward",
                      tolerance=pd.Timedelta(days=4))
        if date_map.Market_Regime.isna().any():
            raise ValueError("Regime not known on some feature dates")
        df = df.merge(date_map[["Date", "Market_Regime"]], on="Date", validate="many_to_one")

    # Reuse audit-verified training labels rather than S03 labels blindly.
    lookup = pd.read_csv(a.label_lookup, low_memory=False,
                         usecols=["Date", "Ticker", "Label_End_Date", "Exact_Target_60d", "Exact_Forward_Return_60d"])
    lookup["Date"] = pd.to_datetime(lookup.Date, errors="raise").dt.normalize()
    lookup["Label_End_Date"] = pd.to_datetime(lookup.Label_End_Date, errors="coerce").dt.normalize()
    if lookup.duplicated(["Ticker", "Date"]).any():
        raise ValueError("Duplicate audited label keys")
    train = df.loc[df.Date.lt(BOUNDARY)].merge(lookup, on=["Date", "Ticker"], how="left", validate="one_to_one")
    good = train.Label_End_Date.notna() & train.Label_End_Date.lt(BOUNDARY)
    LOG.info("Audited train coverage %d/%d", int(good.sum()), len(train))
    train = train.loc[good].copy()
    if train.empty or (train.Label_End_Date <= train.Date).any():
        raise ValueError("Invalid training label maturity dates")
    y = pd.to_numeric(train.Target_60d, errors="coerce")
    exact_y = pd.to_numeric(train.Exact_Target_60d, errors="coerce")
    ret = pd.to_numeric(train.Forward_Return_60d, errors="coerce")
    exact_ret = pd.to_numeric(train.Exact_Forward_Return_60d, errors="coerce")
    if y.isna().any() or exact_y.isna().any() or not y.eq(exact_y).all():
        raise ValueError("Training label disagrees with source-price audit")
    if ret.isna().any() or exact_ret.isna().any() or not np.allclose(ret, exact_ret, atol=1e-7, rtol=0):
        raise ValueError("Training forward return disagrees with source-price audit")
    train["Target_60d"] = y.astype("int8")

    # Use the complete eligible-ticker chronology to determine whether test outcomes
    # have matured. Never turn missing/immature labels into zero.
    df = df.sort_values(["Ticker", "Date"]).copy()
    df["Computed_Label_End_Date"] = df.groupby("Ticker").Date.shift(-HORIZON)
    test = df.loc[df.Date.ge(BOUNDARY)].copy()
    test["Outcome_Mature"] = test.Computed_Label_End_Date.notna()
    # The feature CSV's target is checked only for mature observations.
    actual_return = pd.to_numeric(test.Forward_Return_60d, errors="coerce")
    actual_y = pd.to_numeric(test.Target_60d, errors="coerce")
    valid = test.Outcome_Mature
    if (actual_return[valid].isna().any() or actual_y[valid].isna().any()
        or not actual_y[valid].isin([0, 1]).all()
        or not (actual_y[valid] == (actual_return[valid] > 0).astype(int)).all()):
        raise ValueError("Mature test outcomes inconsistent or missing; check S03")
    test.loc[~valid, ["Target_60d", "Forward_Return_60d"]] = np.nan
    # Stronger maturity check: training uses source-price audit; test maturity
    # is inferred from feature chronology, not independently price-audited.

    exclude = {"Date", "Ticker", "Asset Class", "Category", "Source", "Instrument", "Currency",
               "Market_Regime", "Label_End_Date", "Computed_Label_End_Date", "Outcome_Mature",
               "MA_20d", "MA_50d", "MA_200d", "Exact_Target_60d", "Exact_Forward_Return_60d"}
    suspicious = [c for c in df.columns if c not in exclude and
                  any(k in c.lower() for k in ("future", "lookahead", "next_", "label_", "exit_", "realized_forward"))]
    if suspicious:
        raise ValueError(f"Potential future-information columns: {suspicious}")
    feats = [c for c in df.columns if c not in exclude and not c.startswith(("Target_", "Forward_"))
             and pd.api.types.is_numeric_dtype(df[c])]
    if not feats:
        raise ValueError("No numeric features")
    for part in (train, test):
        for col in feats:
            part[col] = pd.to_numeric(part[col], errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float32")
    LOG.info("Train=%d rows; test=%d rows (%d mature); features=%d; last train maturity=%s",
             len(train), len(test), int(test.Outcome_Mature.sum()), len(feats), train.Label_End_Date.max().date())
    return train, test, feats



def make_preprocessor(features, train_frame, scaled, use_regime):
    """BOTH ticker and asset class are one-hot features, as requested.

    Class is deterministically linked to ticker in this universe (redundant columns);
    regularized linear models and trees therefore need careful interpretation.
    """
    numeric = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scaled:
        numeric.append(("scale", StandardScaler()))
    cols = [("numeric", Pipeline(numeric), features),
            ("ticker", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Ticker"]),
            ("asset_class", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Asset Class"])]
    if use_regime:
        cols.append(("regime", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Market_Regime"]))
    return ColumnTransformer(cols, remainder="drop", sparse_threshold=0)


# FIXED search spaces. Adjusting them is a new experiment, never a modification
# of baseline/metrics or use of post-2022 outcomes for tuning.
SEARCH_SPACES = {
    "LogisticRegression": {
        "C": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2,
              0.1, 0.3, 1, 3, 10, 30, 100, 300],
        "class_weight": [None, "balanced"],
        "tol": [1e-2, 1e-3, 1e-4, 1e-5],
    },
    # Runtime-conscious Random Forest search. The previous space allowed up to
    # 1,200 trees, unbounded depth, tiny leaves, and bootstrap=False. Across
    # every annual fold, those combinations made each trial prohibitively slow.
    # We retain meaningful complexity/regularization choices while bounding the
    # most expensive configurations. All annual folds are still evaluated.
    "RandomForest": {
        "n_estimators": [80, 120, 180, 250, 350],
        "max_depth": [4, 6, 8, 12, 16],
        "min_samples_split": [10, 25, 50, 100],
        "min_samples_leaf": [10, 25, 50, 100],
        "max_features": ["sqrt", 0.2, 0.4, 0.6],
        "max_samples": [0.4, 0.6, 0.8],
        "bootstrap": [True],
        "class_weight": [None, "balanced", "balanced_subsample"],
        "criterion": ["gini", "entropy", "log_loss"],
    },
    "HistGradientBoosting": {
        "max_iter": [100, 180, 300, 450, 650, 900],
        "max_leaf_nodes": [7, 15, 31, 63, 127],
        "max_depth": [3, 5, 7, 10, None],
        "learning_rate": [0.005, 0.01, 0.02, 0.04, 0.07, 0.1, 0.15],
        "min_samples_leaf": [10, 20, 40, 80, 160, 320],
        "l2_regularization": [0.0, 0.01, 0.1, 1.0, 10.0, 100.0],
        "max_bins": [63, 127, 255],
    },
    "XGBoost": {
        "n_estimators": [150, 250, 400, 600, 900, 1300, 1800],
        "max_depth": [2, 3, 4, 5, 6, 8, 10],
        "learning_rate": [0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.15],
        "min_child_weight": [1, 2, 5, 10, 20, 40, 80],
        "subsample": [0.5, 0.65, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.4, 0.55, 0.7, 0.85, 1.0],
        "colsample_bylevel": [0.6, 0.8, 1.0],
        "gamma": [0.0, 0.1, 0.5, 1.0, 3.0, 10.0],
        "reg_lambda": [0.1, 1, 3, 10, 30, 100],
        "reg_alpha": [0, 0.01, 0.1, 1, 5, 20],
        "max_delta_step": [0, 1, 3, 5],
    },
    "CatBoost": {
        "iterations": [150, 250, 400, 600, 900, 1300, 1800],
        "depth": [3, 4, 5, 6, 7, 8, 9, 10],
        "learning_rate": [0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.15],
        "l2_leaf_reg": [1, 3, 5, 10, 20, 40, 80, 150],
        "random_strength": [0.0, 0.1, 0.5, 1, 2, 5, 10],
        "bagging_temperature": [0, 0.2, 0.5, 1, 2, 5],
        "border_count": [32, 64, 128, 254],
        "rsm": [0.5, 0.7, 0.85, 1.0],
    },
}
REFERENCE_PARAMS = {
    "LogisticRegression": {"C": 0.1, "class_weight": None},
    "RandomForest": {"n_estimators": 120, "max_depth": 10, "min_samples_split": 25,
                     "min_samples_leaf": 35, "max_features": "sqrt", "max_samples": 0.6,
                     "bootstrap": True, "class_weight": None},
    "HistGradientBoosting": {"max_iter": 180, "max_leaf_nodes": 15,
                             "learning_rate": 0.05, "min_samples_leaf": 40, "l2_regularization": 0.0},
    "XGBoost": {"n_estimators": 250, "max_depth": 3, "learning_rate": 0.04,
                "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8,
                "reg_lambda": 1.0, "reg_alpha": 0.0},
    "CatBoost": {"iterations": 250, "depth": 5, "learning_rate": 0.04,
                 "l2_leaf_reg": 5, "random_strength": 1.0},
}


def get_estimator(name, workers, seed, params):
    if name == "LogisticRegression":
        # Stable, L2-regularized logistic classifier. Unlike the previous slow
        # elastic-net saga configuration this avoids unbounded convergence times.
        return LogisticRegression(solver="lbfgs", max_iter=600, random_state=seed, **params)
    if name == "RandomForest":
        return RandomForestClassifier(n_jobs=workers, random_state=seed, **params)
    if name == "HistGradientBoosting":
        return HistGradientBoostingClassifier(random_state=seed, **params)
    if name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                             tree_method="hist", n_jobs=workers, random_state=seed, **params)
    if name == "CatBoost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(loss_function="Logloss", verbose=False,
                                  thread_count=workers, random_seed=seed,
                                  allow_writing_files=False, **params)
    raise ValueError(name)


def frequency_predict(fit, evaluation):
    """Frozen baseline, not optimized or modified between runs."""
    y = fit.Target_60d.to_numpy(dtype=np.int8)
    history = fit.groupby("Ticker").Target_60d.agg(["sum", "count"])
    p = ((history["sum"] + 1.0) / (history["count"] + 2.0)).to_dict()
    global_prior = float((y.sum() + 1.0) / (len(y) + 2.0))
    return evaluation.Ticker.map(p).fillna(global_prior).to_numpy(dtype=float)


def transformed_fold(fit, evaluation, features, name, a):
    columns = features + ["Ticker", "Asset Class"] + (["Market_Regime"] if a.with_regime else [])
    transform = make_preprocessor(features, fit, name == "LogisticRegression", a.with_regime)
    x = transform.fit_transform(fit[columns]).astype("float32", copy=False)
    z = transform.transform(evaluation[columns]).astype("float32", copy=False)
    return x, fit.Target_60d.to_numpy(dtype=np.int8), z


def fit_probabilities(name, fit, evaluation, features, a, params):
    if name == "HistoricalFrequency":
        return frequency_predict(fit, evaluation)
    if fit.Target_60d.nunique() < 2:
        raise ValueError(f"{name}: training set contains only one class")
    x, y, z = transformed_fold(fit, evaluation, features, name, a)
    model = get_estimator(name, a.workers, a.seed, params)
    with warnings.catch_warnings(record=True) as warns:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x, y)
    if any(issubclass(w.category, ConvergenceWarning) for w in warns):
        raise RuntimeError(f"{name} failed to converge; configuration={params}")
    return np.clip(np.asarray(model.predict_proba(z)[:, 1], dtype=float), 0, 1)


def validation_folds(training, min_train_years):
    """Exhaustive eligible annual walk-forward blocks before 2023, no handpicked years.

    Each fold trains on all earlier OBSERVABLE labels, including the exact
    audited end date. The earliest calendar year present is treated as initial
    history; require min_train_years COMPLETE years before the first fold.
    """
    start_year = int(training.Date.min().year)
    first_year = start_year + min_train_years + 1
    folds = []
    for year in range(first_year, int(BOUNDARY.year)):
        start, end = pd.Timestamp(year, 1, 1), pd.Timestamp(year+1, 1, 1)
        fit = training.loc[(training.Date < start) & (training.Label_End_Date < start)].copy()
        valid = training.loc[(training.Date >= start) & (training.Date < end)].copy()
        if fit.empty or valid.empty or fit.Target_60d.nunique() < 2:
            continue
        valid = valid.loc[valid.Ticker.isin(fit.Ticker.unique())].copy()
        if valid.empty:
            continue
        if fit.Label_End_Date.max() >= valid.Date.min():
            raise ValueError(f"Invalid temporal / mature label boundary for {year}")
        folds.append((year, fit, valid))
    if not folds:
        raise ValueError("No eligible chronological tuning folds: check --min-train-years and data coverage")
    return folds


# Frozen lexicographic ranking for BOTH tuning and final result reporting.
# PR_AUC means average precision (sklearn.average_precision_score), which is
# a non-interpolated summary of the precision-recall curve.
METRIC_PRIORITY = ("ROC_AUC", "MCC", "Balanced_Accuracy", "PR_AUC", "Log_Loss", "F1")


def metric_sort_tuple(m):
    """Higher is better, except lower log loss; undefined metrics rank last."""
    keys = []
    for col in METRIC_PRIORITY:
        value = m[col]
        if pd.isna(value):
            keys.append(float("-inf"))
        else:
            keys.append(-float(value) if col == "Log_Loss" else float(value))
    return tuple(keys)


def generate_configs(name, trials, seed):
    if name == "HistoricalFrequency":
        return [None]
    # Include the reference configuration as trial 1, so tuning explicitly
    # tests improvement relative to the original untuned model.
    sampled = list(ParameterSampler(SEARCH_SPACES[name], n_iter=max(1, trials-1), random_state=seed))
    configs = [REFERENCE_PARAMS[name]]
    for x in sampled:
        if x not in configs:
            configs.append(x)
    return configs


def tune_model(name, folds, features, a):
    configs = generate_configs(name, a.trials, a.seed)
    best_metric, best_config, best_preds = None, None, None
    trials_info = []
    # Preprocessing is cached once per chronological fold, reused over candidate
    # configurations. Neither imputation nor scaling touches future data.
    cached = []
    if name != "HistoricalFrequency":
        for year, fit, valid in folds:
            if fit.Target_60d.nunique() < 2:
                raise ValueError(f"{name} one-class training period {year}")
            x, y, z = transformed_fold(fit, valid, features, name, a)
            cached.append((x, y, z))
    for trial_index, params in enumerate(configs, start=1):
        trial_preds, fold_metrics = [], []
        begun = time.monotonic()
        try:
            for j, (year, fit, valid) in enumerate(folds):
                if name == "HistoricalFrequency":
                    p = frequency_predict(fit, valid)
                else:
                    x, y, z = cached[j]
                    model = get_estimator(name, a.workers, a.seed, params)
                    with warnings.catch_warnings(record=True) as warns:
                        warnings.simplefilter("always", ConvergenceWarning)
                        model.fit(x, y)
                    if any(issubclass(w.category, ConvergenceWarning) for w in warns):
                        raise RuntimeError("Optimizer did not converge")
                    p = np.clip(np.asarray(model.predict_proba(z)[:, 1], dtype=float), 0, 1)
                part = valid[["Date", "Ticker", "Asset Class", "Target_60d"]].copy()
                part["Model"] = name
                part["Validation_Year"] = year
                part["P_Positive_60d"] = p
                trial_preds.append(part)
                fold_metrics.append({"Model": name, "Trial": trial_index,
                                     "Validation_Year": year, **score(valid.Target_60d, p)})
            combined = pd.concat(trial_preds, ignore_index=True)
            m = score(combined.Target_60d, combined.P_Positive_60d)
            row = {"Model": name, "Trial": trial_index, "Status": "ok",
                   "Parameters": json.dumps(params, sort_keys=True),
                   **m, "Seconds": round(time.monotonic()-begun, 2)}
            trials_info.append(row)
            LOG.info("  tuning %s trial %d/%d | AUC=%.4f MCC=%.4f balanced_acc=%.4f PR_AUC=%.4f logloss=%.4f F1=%.4f | %.1fs",
                     name, trial_index, len(configs), m["ROC_AUC"], m["MCC"],
                     m["Balanced_Accuracy"], m["PR_AUC"], m["Log_Loss"], m["F1"], row["Seconds"])
            if best_metric is None or metric_sort_tuple(m) > metric_sort_tuple(best_metric):
                best_metric, best_config = m, params
                best_preds = (combined, fold_metrics)
        except Exception as e:
            LOG.warning("  FAILED %s trial %d/%d params=%s reason=%s", name, trial_index, len(configs), params, e)
            trials_info.append({"Model": name, "Trial": trial_index, "Status": "failed",
                                "Parameters": json.dumps(params, sort_keys=True), "Error": str(e),
                                "Seconds": round(time.monotonic()-begun, 2)})
    if best_metric is None:
        raise RuntimeError(f"All hyperparameter trials failed for {name}")
    LOG.info("  SELECTED %s | %s | validation ROC_AUC=%.4f MCC=%.4f", name,
             json.dumps(best_config, sort_keys=True), best_metric["ROC_AUC"], best_metric["MCC"])
    return best_config, best_preds, trials_info

def save_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def ranking(metric_frame):
    """Rank using the SAME frozen priority used for hyperparameter selection.

    ROC-AUC > MCC > Balanced Accuracy > PR-AUC > lower Log Loss > F1.
    This is a lexicographic ordering, not an arbitrary weighted score.
    """
    if metric_frame.empty:
        raise ValueError("No model metrics to rank")
    cols = list(METRIC_PRIORITY)
    ranks = metric_frame[["Model", *cols]].copy()
    ranks = ranks.sort_values(cols + ["Model"],
                              ascending=[True if c == "Log_Loss" else False for c in cols] + [True],
                              na_position="last", kind="stable").reset_index(drop=True)
    ranks.insert(0, "Rank", np.arange(1, len(ranks) + 1))
    return ranks


def run(a):
    if a.workers < 1 or not a.models or len(set(a.models)) != len(a.models):
        raise ValueError("Invalid workers or repeated/empty model list")
    if a.trials < 2 or a.min_train_years < 1:
        raise ValueError("--trials must be >=2 and --min-train-years >=1")
    a.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(a.out / "run.log", encoding="utf8")], force=True)
    training, testing, features = load_data(a)
    folds = validation_folds(training, a.min_train_years)
    fold_years = [year for year, _, _ in folds]
    LOG.info("ALL ELIGIBLE PRE-2023 ANNUAL TUNING FOLDS=%s; configurations/model=%s; fixed threshold=%.2f; rank=%s",
             fold_years, a.trials, THRESHOLD, " > ".join(METRIC_PRIORITY))
    LOG.info("Expected ML fits: %d trials x %d annual folds x %d selected ML models = %d historical fits, plus final fits",
             a.trials, len(folds), sum(n != "HistoricalFrequency" for n in a.models),
             a.trials * len(folds) * sum(n != "HistoricalFrequency" for n in a.models))
    LOG.info("RandomForest tuning is runtime-bounded: 80-350 trees, bootstrap subsampling max_samples=0.4-0.8, bounded depth/leaf sizes; annual folds retained")
    validation_preds, test_preds, selected_fold_metrics, all_trials = [], [], [], []
    selected_params = {}
    for name in a.models:
        LOG.info("MODEL %s", name)
        params, (valpart, foldmetrics), trials = tune_model(name, folds, features, a)
        selected_params[name] = params
        all_trials.extend(trials)
        selected_fold_metrics.extend(foldmetrics)
        validation_preds.append(valpart)
        ptest = fit_probabilities(name, training, testing, features, a, params)
        part = testing[["Date", "Ticker", "Asset Class", "Outcome_Mature", "Computed_Label_End_Date",
                        "Target_60d", "Forward_Return_60d"]].copy()
        part["Model"] = name
        part["P_Positive_60d"] = ptest
        part["Prediction_0_5"] = (ptest >= THRESHOLD).astype("int8")
        test_preds.append(part)
        scored = part.loc[part.Outcome_Mature]
        m = score(scored.Target_60d, scored.P_Positive_60d)
        LOG.info("  2023+ TEST %s | N=%d AUC=%.4f MCC=%.4f balanced_acc=%.4f PR_AUC=%.4f logloss=%.4f F1=%.4f precision=%.4f recall=%.4f",
                 name, m["N"], m["ROC_AUC"], m["MCC"], m["Balanced_Accuracy"],
                 m["PR_AUC"], m["Log_Loss"], m["F1"], m["Precision"], m["Recall"])
    val = pd.concat(validation_preds, ignore_index=True)
    test = pd.concat(test_preds, ignore_index=True)
    if val.duplicated(["Date", "Ticker", "Model"]).any() or test.duplicated(["Date", "Ticker", "Model"]).any():
        raise ValueError("Duplicate prediction keys")
    valmetrics = pd.DataFrame([{"Model": name, **score(g.Target_60d, g.P_Positive_60d)}
                               for name, g in val.groupby("Model")])
    ranking_table = ranking(valmetrics)
    mature = test.loc[test.Outcome_Mature]
    testmetrics = pd.DataFrame([{"Model": name, **score(g.Target_60d, g.P_Positive_60d)}
                                for name, g in mature.groupby("Model")])
    by_class = pd.DataFrame([{"Model": name, "Asset_Class": cls,
                              **score(g.Target_60d, g.P_Positive_60d)}
                             for (name, cls), g in mature.groupby(["Model", "Asset Class"])])
    by_ticker = pd.DataFrame([{"Model": name, "Ticker": ticker,
                               **score(g.Target_60d, g.P_Positive_60d)}
                              for (name, ticker), g in mature.groupby(["Model", "Ticker"])])
    # Test ranking is diagnostic only; never selects/tunes hyperparameters.
    test_ranking = ranking(testmetrics)
    LOG.info("TUNED VALIDATION RANKING (fixed priority):\n%s", ranking_table.to_string(index=False))
    LOG.info("2023+ TEST RANKING (report-only; no selection on test):\n%s", test_ranking.to_string(index=False))
    save_csv(val, a.out / "validation_predictions.csv")
    save_csv(test, a.out / "predictions_2023plus.csv")
    save_csv(pd.DataFrame(selected_fold_metrics), a.out / "validation_metrics_by_year.csv")
    save_csv(valmetrics, a.out / "validation_metrics_pooled.csv")
    save_csv(ranking_table, a.out / "model_ranking_validation.csv")
    save_csv(testmetrics, a.out / "test_metrics_pooled.csv")
    save_csv(test_ranking, a.out / "model_ranking_2023plus_report_only.csv")
    save_csv(by_class, a.out / "test_metrics_by_asset_class.csv")
    save_csv(by_ticker, a.out / "test_metrics_by_ticker.csv")
    save_csv(pd.DataFrame(all_trials), a.out / "hyperparameter_search_results.csv")
    save_csv(pd.DataFrame({"Feature": features}), a.out / "numeric_features.csv")
    manifest = {"design": "one pooled classifier per algorithm; one-hot ticker AND asset class",
                "target": "positive 60-trading-day forward return", "train_label_end_exclusive": "2023-01-01",
                "tune_years": fold_years, "min_train_years": a.min_train_years,
                "requested_trials_per_ml_model": a.trials,
                "test_date_from": "2023-01-01", "threshold": THRESHOLD,
                "baseline": "per-ticker historical positive-return frequency, Beta(1,1) smoothing",
                "metric_rank_priority": ["ROC_AUC desc", "MCC desc", "Balanced_Accuracy desc", "PR_AUC desc", "Log_Loss asc", "F1 desc"],
                "PR_AUC_definition": "average_precision_score: non-interpolated average precision",
                "aggregation": "pooled pre-2023 validation observations, micro weighting",
                "model_selection": "pre-2023 validation only; 2023+ report only",
                "selected_hyperparameters": selected_params,
                "models": a.models, "features": len(features), "train_rows": len(training),
                "test_rows": len(testing), "test_outcome_maturity": "inferred from feature chronology; not source-price-audited",
                "regime_enabled": a.with_regime,
                "warnings": ["Feature availability/lag not independently audited",
                             "2023+ previously inspected in other experiments; not pristine holdout",
                             "Overlapping 60-day target observations are serially dependent",
                             "Asset class is deterministic given ticker in static universe"]}
    (a.out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOG.info("DONE | %d training rows, %d test predictions (%d mature); out=%s",
             len(training), len(test), len(mature), a.out)
    return ranking_table


if __name__ == "__main__":
    run(parse_args())
