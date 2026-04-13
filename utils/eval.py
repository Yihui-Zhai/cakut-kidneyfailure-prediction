import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, brier_score_loss,
    precision_recall_curve, auc
)
from sklearn.linear_model import LogisticRegression
from scipy.stats import beta
from scipy.optimize import minimize

def metric_ci_bootstrap(y_true, y_scores, y_preds, metric_fn, n_bootstraps=1000, alpha=0.95, random_seed=42):
    rng = np.random.RandomState(random_seed)
    bootstrapped_scores = []
    
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_preds = np.array(y_preds)
    n_samples = len(y_true)

    for _ in range(n_bootstraps):
        indices = rng.randint(0, n_samples, n_samples)
        y_t = y_true[indices]
        y_s = y_scores[indices]
        y_p = y_preds[indices]
        
        if metric_fn.__name__ == 'roc_auc_score':
            if len(np.unique(y_t)) < 2:
                continue
            score = metric_fn(y_t, y_s)
        else:
            score = metric_fn(y_t, y_p)

        bootstrapped_scores.append(score)

    sorted_scores = np.sort(bootstrapped_scores)
    lower = np.percentile(sorted_scores, (1 - alpha) / 2 * 100)
    upper = np.percentile(sorted_scores, (alpha + (1 - alpha) / 2) * 100)
    mean = np.mean(bootstrapped_scores)

    if metric_fn.__name__ == 'roc_auc_score':
        return metric_fn(y_true, y_scores), lower, upper
    return metric_fn(y_true, y_preds), lower, upper


def auprc_ci_bootstrap(y_true, y_scores, n_bootstraps=1000, alpha=0.95, random_seed=42):
    rng = np.random.RandomState(random_seed)
    bootstrapped_scores = []
    
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    n_samples = len(y_true)

    for _ in range(n_bootstraps):
        indices = rng.randint(0, n_samples, n_samples)
        y_t = y_true[indices]
        y_s = y_scores[indices]

        if len(np.unique(y_t)) < 2:
            continue

        precision, recall, _ = precision_recall_curve(y_t, y_s)
        score = auc(recall, precision)
        bootstrapped_scores.append(score)

    sorted_scores = np.sort(bootstrapped_scores)
    lower = np.percentile(sorted_scores, (1 - alpha) / 2 * 100)
    upper = np.percentile(sorted_scores, (alpha + (1 - alpha) / 2) * 100)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    score = auc(recall, precision)

    return score, lower, upper


def score_ci_bootstrap(y_true, y_scores, score_fn, n_bootstraps=1000, alpha=0.95, random_seed=42, require_two_classes=False):
    rng = np.random.RandomState(random_seed)
    bootstrapped_scores = []

    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    n_samples = len(y_true)

    for _ in range(n_bootstraps):
        indices = rng.randint(0, n_samples, n_samples)
        y_t = y_true[indices]
        y_s = y_scores[indices]

        if require_two_classes and len(np.unique(y_t)) < 2:
            continue

        try:
            score = score_fn(y_t, y_s)
        except Exception:
            continue
        if np.isfinite(score):
            bootstrapped_scores.append(score)

    try:
        point = score_fn(y_true, y_scores)
    except Exception:
        point = np.nan

    if len(bootstrapped_scores) == 0:
        return point, np.nan, np.nan

    # If the full-sample estimate fails but bootstrap estimates exist,
    # fall back to a robust summary of the bootstrap distribution.
    if not np.isfinite(point):
        point = float(np.nanmedian(bootstrapped_scores))

    sorted_scores = np.sort(bootstrapped_scores)
    lower = np.percentile(sorted_scores, (1 - alpha) / 2 * 100)
    upper = np.percentile(sorted_scores, (alpha + (1 - alpha) / 2) * 100)
    return point, lower, upper


def _sigmoid(x):
    x = np.clip(x, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x))


def _to_lp_from_scores(y_scores):
    """
    Convert predicted probabilities to linear predictor (LP) via logit.
    """
    y_scores = np.asarray(y_scores, dtype=float)
    eps = 1e-15
    p = np.clip(y_scores, eps, 1 - eps)
    return np.log(p / (1 - p))


def _calibration_intercept_slope(y_true, y_lp):
    """
    Compute calibration intercept and slope using literature-style definitions.

    - Calibration-in-the-large (intercept): fit alpha with slope fixed at 1
      logit(P(Y=1)) = alpha + 1 * LP
    - Calibration slope: fit alpha and beta
      logit(P(Y=1)) = alpha + beta * LP

    LP should be the model's linear predictor (raw score / log-odds scale).
    """
    y = np.asarray(y_true, dtype=float)
    lp = np.asarray(y_lp, dtype=float).reshape(-1)

    if y.size == 0:
        return np.nan, np.nan
    if y.size != lp.size:
        return np.nan, np.nan

    # 1) Calibration intercept (CITL): beta fixed at 1, estimate alpha only.
    def nll_alpha(theta):
        alpha = theta[0]
        eta = alpha + lp
        p = _sigmoid(eta)
        return -np.sum(y * np.log(p + 1e-15) + (1 - y) * np.log(1 - p + 1e-15))

    res_alpha = minimize(
        nll_alpha,
        x0=np.array([0.0], dtype=float),
        method="BFGS",
    )
    intercept = float(res_alpha.x[0]) if res_alpha.success else np.nan
    if not np.isfinite(intercept):
        # Robust fallback: solve score equation sum(y - sigmoid(alpha+lp)) = 0.
        # Monotone in alpha; bracket wide to avoid overflow (sigmoid clips internally).
        def g(alpha):
            p = _sigmoid(alpha + lp)
            return float(np.sum(y - p))

        # If classes are degenerate, CITL isn't identifiable in finite terms.
        if np.unique(y).size >= 2:
            try:
                from scipy.optimize import brentq

                # Expand bracket if needed.
                lo, hi = -50.0, 50.0
                glo, ghi = g(lo), g(hi)
                if glo == 0.0:
                    intercept = lo
                elif ghi == 0.0:
                    intercept = hi
                elif glo * ghi < 0:
                    intercept = float(brentq(g, lo, hi, maxiter=200))
            except Exception:
                pass

    # 2) Calibration slope: estimate alpha and beta jointly.
    def nll_alpha_beta(theta):
        alpha, beta_coef = theta
        eta = alpha + beta_coef * lp
        p = _sigmoid(eta)
        return -np.sum(y * np.log(p + 1e-15) + (1 - y) * np.log(1 - p + 1e-15))

    res_ab = minimize(
        nll_alpha_beta,
        x0=np.array([0.0, 1.0], dtype=float),
        method="BFGS",
    )
    slope = float(res_ab.x[1]) if res_ab.success else np.nan
    if not np.isfinite(slope):
        # Robust fallback: regularized logistic regression of y on lp.
        # Use large C to approximate unpenalized fit while keeping numerical stability.
        try:
            X = lp.reshape(-1, 1)
            if np.unique(y).size >= 2 and np.isfinite(X).all():
                lr = LogisticRegression(
                    penalty="l2",
                    C=1e6,
                    solver="lbfgs",
                    fit_intercept=True,
                    max_iter=2000,
                )
                lr.fit(X, y.astype(int))
                slope = float(lr.coef_.reshape(-1)[0])
                # Note: intercept for the slope model isn't returned by this function;
                # callers use slope and intercept separately.
        except Exception:
            pass

    return intercept, slope


def calibration_intercept(y_true, y_lp):
    intercept, _ = _calibration_intercept_slope(y_true, y_lp)
    return intercept


def calibration_slope(y_true, y_lp):
    _, slope = _calibration_intercept_slope(y_true, y_lp)
    return slope


def _extract_linear_predictor(model, X, y_pred_prob=None):
    """
    Get model LP (linear predictor) for calibration.
    Priority:
      1) decision_function
      2) CatBoost RawFormulaVal
      3) log(p1/p0) from predict_log_proba
      4) fallback logit(predicted_probability)
    """
    if hasattr(model, "decision_function"):
        try:
            lp = np.asarray(model.decision_function(X), dtype=float).reshape(-1)
            return lp
        except Exception:
            pass

    if hasattr(model, "predict"):
        try:
            lp = model.predict(X, prediction_type="RawFormulaVal")
            lp = np.asarray(lp, dtype=float).reshape(-1)
            return lp
        except Exception:
            pass

    if hasattr(model, "predict_log_proba"):
        try:
            log_proba = np.asarray(model.predict_log_proba(X), dtype=float)
            if log_proba.ndim == 2 and log_proba.shape[1] >= 2:
                return (log_proba[:, 1] - log_proba[:, 0]).reshape(-1)
        except Exception:
            pass

    if y_pred_prob is None:
        y_pred_prob = model.predict_proba(X)[:, 1]
    return _to_lp_from_scores(y_pred_prob)


def _plr_from_cm(tn, fp, fn, tp):
    """PLR = sensitivity / (1 - specificity) = sens / FPR. Undefined when FPR == 0."""
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    d = tn + fp
    if d == 0:
        return np.nan
    fpr = fp / d
    if fpr == 0:
        return np.inf if sens > 0 else 0.0
    return sens / fpr


def _nlr_from_cm(tn, fp, fn, tp):
    """NLR = (1 - sensitivity) / specificity = FNR / specificity. Undefined when specificity == 0."""
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    d = tn + fp
    if d == 0:
        return np.nan
    spec = tn / d
    if spec == 0:
        return np.inf if fnr > 0 else 0.0
    return fnr / spec


def _fmt_float_ci(mean, lo, hi):
    def fmt(v):
        v = float(v)
        if np.isnan(v):
            return "nan"
        if np.isinf(v):
            return "inf" if v > 0 else "-inf"
        return f"{v:.3f}"

    return f"{fmt(mean)} ({fmt(lo)}-{fmt(hi)})"


def clopper_pearson_ci(k, n, alpha=0.95):
    """Exact (Clopper-Pearson) CI for binomial proportion k/n."""
    if n <= 0:
        return np.nan, np.nan
    tail = (1 - alpha) / 2
    lower = 0.0 if k == 0 else beta.ppf(tail, k, n - k + 1)
    upper = 1.0 if k == n else beta.ppf(1 - tail, k + 1, n - k)
    return float(lower), float(upper)


def fallback_ci_to_cp(point, lower, upper, k, n, alpha=0.95):
    """Use CP interval when bootstrap CI becomes nan/inf."""
    if np.isfinite(lower) and np.isfinite(upper):
        return point, lower, upper
    cp_lower, cp_upper = clopper_pearson_ci(k, n, alpha=alpha)
    if np.isfinite(point):
        cp_point = point
    else:
        cp_point = (k / n) if n > 0 else np.nan
    return cp_point, cp_lower, cp_upper


def plr_ci_bootstrap(y_true, y_preds, n_bootstraps=1000, alpha=0.95, random_seed=42):
    """Bootstrap CI for PLR; skips resamples where FPR == 0 (ratio undefined / inf)."""
    rng = np.random.RandomState(random_seed)
    bootstrapped_scores = []
    y_true = np.asarray(y_true)
    y_preds = np.asarray(y_preds)
    n_samples = len(y_true)

    for _ in range(n_bootstraps):
        indices = rng.randint(0, n_samples, n_samples)
        y_t = y_true[indices]
        y_p = y_preds[indices]
        cm = confusion_matrix(y_t, y_p, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        d = tn + fp
        if d == 0:
            continue
        fpr = fp / d
        if fpr == 0:
            continue
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        bootstrapped_scores.append(sens / fpr)

    cm0 = confusion_matrix(y_true, y_preds, labels=[0, 1])
    tn, fp, fn, tp = cm0.ravel()
    point = _plr_from_cm(tn, fp, fn, tp)

    if len(bootstrapped_scores) == 0:
        return point, np.nan, np.nan

    arr = np.sort(bootstrapped_scores)
    lower = np.percentile(arr, (1 - alpha) / 2 * 100)
    upper = np.percentile(arr, (alpha + (1 - alpha) / 2) * 100)
    return point, lower, upper


def nlr_ci_bootstrap(y_true, y_preds, n_bootstraps=1000, alpha=0.95, random_seed=42):
    """Bootstrap CI for NLR; skips resamples where specificity == 0."""
    rng = np.random.RandomState(random_seed)
    bootstrapped_scores = []
    y_true = np.asarray(y_true)
    y_preds = np.asarray(y_preds)
    n_samples = len(y_true)

    for _ in range(n_bootstraps):
        indices = rng.randint(0, n_samples, n_samples)
        y_t = y_true[indices]
        y_p = y_preds[indices]
        cm = confusion_matrix(y_t, y_p, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        d = tn + fp
        if d == 0:
            continue
        spec = tn / d
        if spec == 0:
            continue
        fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
        bootstrapped_scores.append(fnr / spec)

    cm0 = confusion_matrix(y_true, y_preds, labels=[0, 1])
    tn, fp, fn, tp = cm0.ravel()
    point = _nlr_from_cm(tn, fp, fn, tp)

    if len(bootstrapped_scores) == 0:
        return point, np.nan, np.nan

    arr = np.sort(bootstrapped_scores)
    lower = np.percentile(arr, (1 - alpha) / 2 * 100)
    upper = np.percentile(arr, (alpha + (1 - alpha) / 2) * 100)
    return point, lower, upper


def eval_model(model, X_test, y_test, n_bootstraps=1000, calibration_lp=None):
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # Use the estimator's native decision rule for class predictions when possible.
    # This matches SVM's `decision_function >= 0` threshold (i.e., `predict()`),
    # which can differ from `predict_proba >= 0.5` due to probability calibration.
    try:
        y_pred = np.asarray(model.predict(X_test), dtype=int).reshape(-1)
    except Exception:
        y_pred = (y_pred_prob >= 0.5).astype(int)
    if calibration_lp is None:
        lp_for_calibration = _extract_linear_predictor(model, X_test, y_pred_prob=y_pred_prob)
    else:
        lp_for_calibration = np.asarray(calibration_lp, dtype=float).reshape(-1)
        if lp_for_calibration.size != len(y_test):
            raise ValueError("Length of calibration_lp must match y_test.")
    
    auc_mean, auc_lower, auc_upper = metric_ci_bootstrap(
        y_test, y_pred_prob, y_pred, roc_auc_score, n_bootstraps
    )

    auprc_mean, auprc_lower, auprc_upper = auprc_ci_bootstrap(
        y_test, y_pred_prob, n_bootstraps
    )

    acc_mean, acc_lower, acc_upper = metric_ci_bootstrap(
        y_test, y_pred_prob, y_pred, accuracy_score, n_bootstraps
    )

    prec_mean, prec_lower, prec_upper = metric_ci_bootstrap(
        y_test, y_pred_prob, y_pred, precision_score, n_bootstraps
    )

    rec_mean, rec_lower, rec_upper = metric_ci_bootstrap(
        y_test, y_pred_prob, y_pred, recall_score, n_bootstraps
    )

    f1_mean, f1_lower, f1_upper = metric_ci_bootstrap(
        y_test, y_pred_prob, y_pred, f1_score, n_bootstraps
    )

    cm_full = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm_full.ravel()

    # specificity
    def specificity_fn(y_t, y_p):
        cm = confusion_matrix(y_t, y_p, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0

    spec_mean, spec_lower, spec_upper = metric_ci_bootstrap(
        y_test, y_pred_prob, y_pred, specificity_fn, n_bootstraps
    )
    spec_mean, spec_lower, spec_upper = fallback_ci_to_cp(
        spec_mean, spec_lower, spec_upper, k=tn, n=tn + fp
    )

    # PPV = TP / (TP + FP); NPV = TN / (TN + FN)
    def ppv_fn(y_t, y_p):
        cm = confusion_matrix(y_t, y_p, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        d = tp + fp
        return tp / d if d > 0 else 0.0

    def npv_fn(y_t, y_p):
        cm = confusion_matrix(y_t, y_p, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        d = tn + fn
        return tn / d if d > 0 else 0.0

    ppv_mean, ppv_lower, ppv_upper = metric_ci_bootstrap(
        y_test, y_pred_prob, y_pred, ppv_fn, n_bootstraps
    )
    ppv_mean, ppv_lower, ppv_upper = fallback_ci_to_cp(
        ppv_mean, ppv_lower, ppv_upper, k=tp, n=tp + fp
    )
    npv_mean, npv_lower, npv_upper = metric_ci_bootstrap(
        y_test, y_pred_prob, y_pred, npv_fn, n_bootstraps
    )
    npv_mean, npv_lower, npv_upper = fallback_ci_to_cp(
        npv_mean, npv_lower, npv_upper, k=tn, n=tn + fn
    )
    plr_mean, plr_lower, plr_upper = plr_ci_bootstrap(
        y_test, y_pred, n_bootstraps
    )
    nlr_mean, nlr_lower, nlr_upper = nlr_ci_bootstrap(
        y_test, y_pred, n_bootstraps
    )
    brier_mean, brier_lower, brier_upper = score_ci_bootstrap(
        y_test, y_pred_prob, brier_score_loss, n_bootstraps
    )
    cal_slope_mean, cal_slope_lower, cal_slope_upper = score_ci_bootstrap(
        y_test, lp_for_calibration, calibration_slope, n_bootstraps, require_two_classes=True
    )
    cal_intercept_mean, cal_intercept_lower, cal_intercept_upper = score_ci_bootstrap(
        y_test, lp_for_calibration, calibration_intercept, n_bootstraps, require_two_classes=True
    )

    return {
        "AUC": f"{auc_mean:.3f} ({auc_lower:.3f}-{auc_upper:.3f})",
        "AUPRC": f"{auprc_mean:.3f} ({auprc_lower:.3f}-{auprc_upper:.3f})",
        "Accuracy": f"{acc_mean:.3f} ({acc_lower:.3f}-{acc_upper:.3f})",
        "Precision": f"{prec_mean:.3f} ({prec_lower:.3f}-{prec_upper:.3f})",
        "Recall": f"{rec_mean:.3f} ({rec_lower:.3f}-{rec_upper:.3f})",
        "Specificity": f"{spec_mean:.3f} ({spec_lower:.3f}-{spec_upper:.3f})",
        "F1": f"{f1_mean:.3f} ({f1_lower:.3f}-{f1_upper:.3f})",
        "PPV": f"{ppv_mean:.3f} ({ppv_lower:.3f}-{ppv_upper:.3f})",
        "NPV": f"{npv_mean:.3f} ({npv_lower:.3f}-{npv_upper:.3f})",
        "PLR": _fmt_float_ci(plr_mean, plr_lower, plr_upper),
        "NLR": _fmt_float_ci(nlr_mean, nlr_lower, nlr_upper),
        "Brier Score": f"{brier_mean:.3f} ({brier_lower:.3f}-{brier_upper:.3f})",
        "Calibration Slope": f"{cal_slope_mean:.3f} ({cal_slope_lower:.3f}-{cal_slope_upper:.3f})",
        "Calibration Intercept": f"{cal_intercept_mean:.3f} ({cal_intercept_lower:.3f}-{cal_intercept_upper:.3f})",
    }


def eval_predictions_binary(
    y_true,
    y_pred_prob,
    threshold: float = 0.5,
    n_bootstraps: int = 1000,
):
    """
    Evaluate binary classifier predictions with bootstrap CIs.

    This mirrors `eval_model` but takes arrays instead of a fitted model.
    Calibration slope/intercept are computed using LP=logit(p).
    """
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_pred_prob = np.asarray(y_pred_prob, dtype=float).reshape(-1)
    y_pred_prob = np.clip(y_pred_prob, 0.0, 1.0)
    y_pred = (y_pred_prob >= float(threshold)).astype(int)

    lp_for_calibration = _to_lp_from_scores(y_pred_prob)

    def _precision(y_t, y_p):
        return precision_score(y_t, y_p, zero_division=0)

    def _recall(y_t, y_p):
        return recall_score(y_t, y_p, zero_division=0)

    def _f1(y_t, y_p):
        return f1_score(y_t, y_p, zero_division=0)

    auc_mean, auc_lower, auc_upper = metric_ci_bootstrap(
        y_true, y_pred_prob, y_pred, roc_auc_score, n_bootstraps
    )
    auprc_mean, auprc_lower, auprc_upper = auprc_ci_bootstrap(
        y_true, y_pred_prob, n_bootstraps
    )
    acc_mean, acc_lower, acc_upper = metric_ci_bootstrap(
        y_true, y_pred_prob, y_pred, accuracy_score, n_bootstraps
    )
    prec_mean, prec_lower, prec_upper = metric_ci_bootstrap(
        y_true, y_pred_prob, y_pred, _precision, n_bootstraps
    )
    rec_mean, rec_lower, rec_upper = metric_ci_bootstrap(
        y_true, y_pred_prob, y_pred, _recall, n_bootstraps
    )
    f1_mean, f1_lower, f1_upper = metric_ci_bootstrap(
        y_true, y_pred_prob, y_pred, _f1, n_bootstraps
    )

    cm_full = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm_full.ravel()

    def specificity_fn(y_t, y_p):
        cm = confusion_matrix(y_t, y_p, labels=[0, 1])
        tn_, fp_, fn_, tp_ = cm.ravel()
        return tn_ / (tn_ + fp_) if (tn_ + fp_) > 0 else 0.0

    spec_mean, spec_lower, spec_upper = metric_ci_bootstrap(
        y_true, y_pred_prob, y_pred, specificity_fn, n_bootstraps
    )
    spec_mean, spec_lower, spec_upper = fallback_ci_to_cp(
        spec_mean, spec_lower, spec_upper, k=tn, n=tn + fp
    )

    def ppv_fn(y_t, y_p):
        cm = confusion_matrix(y_t, y_p, labels=[0, 1])
        tn_, fp_, fn_, tp_ = cm.ravel()
        d = tp_ + fp_
        return tp_ / d if d > 0 else 0.0

    def npv_fn(y_t, y_p):
        cm = confusion_matrix(y_t, y_p, labels=[0, 1])
        tn_, fp_, fn_, tp_ = cm.ravel()
        d = tn_ + fn_
        return tn_ / d if d > 0 else 0.0

    ppv_mean, ppv_lower, ppv_upper = metric_ci_bootstrap(
        y_true, y_pred_prob, y_pred, ppv_fn, n_bootstraps
    )
    ppv_mean, ppv_lower, ppv_upper = fallback_ci_to_cp(
        ppv_mean, ppv_lower, ppv_upper, k=tp, n=tp + fp
    )
    npv_mean, npv_lower, npv_upper = metric_ci_bootstrap(
        y_true, y_pred_prob, y_pred, npv_fn, n_bootstraps
    )
    npv_mean, npv_lower, npv_upper = fallback_ci_to_cp(
        npv_mean, npv_lower, npv_upper, k=tn, n=tn + fn
    )

    plr_mean, plr_lower, plr_upper = plr_ci_bootstrap(
        y_true, y_pred, n_bootstraps
    )
    nlr_mean, nlr_lower, nlr_upper = nlr_ci_bootstrap(
        y_true, y_pred, n_bootstraps
    )
    brier_mean, brier_lower, brier_upper = score_ci_bootstrap(
        y_true, y_pred_prob, brier_score_loss, n_bootstraps
    )
    cal_slope_mean, cal_slope_lower, cal_slope_upper = score_ci_bootstrap(
        y_true,
        lp_for_calibration,
        calibration_slope,
        n_bootstraps,
        require_two_classes=True,
    )
    cal_intercept_mean, cal_intercept_lower, cal_intercept_upper = score_ci_bootstrap(
        y_true,
        lp_for_calibration,
        calibration_intercept,
        n_bootstraps,
        require_two_classes=True,
    )

    return {
        "AUC": f"{auc_mean:.3f} ({auc_lower:.3f}-{auc_upper:.3f})",
        "AUPRC": f"{auprc_mean:.3f} ({auprc_lower:.3f}-{auprc_upper:.3f})",
        "Accuracy": f"{acc_mean:.3f} ({acc_lower:.3f}-{acc_upper:.3f})",
        "Precision": f"{prec_mean:.3f} ({prec_lower:.3f}-{prec_upper:.3f})",
        "Recall": f"{rec_mean:.3f} ({rec_lower:.3f}-{rec_upper:.3f})",
        "Specificity": f"{spec_mean:.3f} ({spec_lower:.3f}-{spec_upper:.3f})",
        "F1": f"{f1_mean:.3f} ({f1_lower:.3f}-{f1_upper:.3f})",
        "PPV": f"{ppv_mean:.3f} ({ppv_lower:.3f}-{ppv_upper:.3f})",
        "NPV": f"{npv_mean:.3f} ({npv_lower:.3f}-{npv_upper:.3f})",
        "PLR": _fmt_float_ci(plr_mean, plr_lower, plr_upper),
        "NLR": _fmt_float_ci(nlr_mean, nlr_lower, nlr_upper),
        "Brier Score": f"{brier_mean:.3f} ({brier_lower:.3f}-{brier_upper:.3f})",
        "Calibration Slope": f"{cal_slope_mean:.3f} ({cal_slope_lower:.3f}-{cal_slope_upper:.3f})",
        "Calibration Intercept": f"{cal_intercept_mean:.3f} ({cal_intercept_lower:.3f}-{cal_intercept_upper:.3f})",
    }


def export_catboost_subgroup_metrics_xlsx(
    cohort_predictions_xlsx: str,
    output_xlsx: str,
    subgroup_col: str = "cakut_subphenotype",
    years=(1, 3, 5),
    threshold: float = 0.5,
    n_bootstraps: int = 1000,
    model_key: str = "catboost",
):
    """
    Generate CatBoost subgroup metrics (per year) from a cohort-level Excel.

    Expected input schema (same as `ml/plot.py`):
    - labels: `esrd_{year}y` in {0,1} with -1 treated as invalid
    - predictions: `{model_key}_pred_{year}y` as probabilities in [0, 1]
    - subgroup column: default `cakut_subphenotype`

    Output:
    - one Excel file with sheets `1y`, `3y`, `5y`
    - each sheet has rows: Overall + each subgroup, with N/Pos/Neg and metrics w/ CI
    """
    df = pd.read_excel(cohort_predictions_xlsx)
    if subgroup_col not in df.columns:
        raise ValueError(f"Missing subgroup column: {subgroup_col}")

    years = list(years)
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        for year in years:
            y_col = f"esrd_{year}y"
            pred_col = f"{model_key}_pred_{year}y"
            missing = [c for c in (y_col, pred_col) if c not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns for year={year}: {missing}")

            valid = df[y_col].notna() & (df[y_col] != -1) & df[pred_col].notna()
            d = df.loc[valid, [subgroup_col, y_col, pred_col]].copy()
            d[y_col] = d[y_col].astype(int)
            d[pred_col] = d[pred_col].astype(float).clip(0, 1)

            rows = []

            def add_block(name, block):
                y_true = block[y_col].to_numpy()
                y_prob = block[pred_col].to_numpy()
                n = int(len(block))
                n_pos = int((y_true == 1).sum())
                n_neg = int((y_true == 0).sum())

                # If a subgroup has only one class, AUC/AUPRC are not defined.
                metrics = (
                    eval_predictions_binary(
                        y_true, y_prob, threshold=threshold, n_bootstraps=n_bootstraps
                    )
                    if n > 0 and np.unique(y_true).size >= 2
                    else {
                        "AUC": "nan",
                        "AUPRC": "nan",
                        "Accuracy": "nan",
                        "Precision": "nan",
                        "Recall": "nan",
                        "Specificity": "nan",
                        "F1": "nan",
                        "PPV": "nan",
                        "NPV": "nan",
                        "PLR": "nan",
                        "NLR": "nan",
                        "Brier Score": "nan",
                        "Calibration Slope": "nan",
                        "Calibration Intercept": "nan",
                    }
                )

                rows.append(
                    {
                        "Subgroup": name,
                        "N": n,
                        "Pos": n_pos,
                        "Neg": n_neg,
                        **metrics,
                    }
                )

            add_block("Overall", d)
            for sg, block in d.groupby(subgroup_col, dropna=False):
                add_block(str(sg), block)

            out = pd.DataFrame(rows)
            out.to_excel(writer, sheet_name=f"{year}y", index=False)


