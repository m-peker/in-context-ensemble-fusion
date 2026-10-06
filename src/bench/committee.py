"""Build base-model committees on OpenML tasks.

For every (dataset, outer fold) we store:
  P_oof : (n_tr, K, C) out-of-fold class probabilities on the training rows (inner CV)
  P_te  : (n_te, K, C) test probabilities from base models refit on the full training rows
  y_tr, y_te, X_tr, X_te (standardized numeric matrix, used only for kNN locality features)
All fusion methods are later evaluated offline on these arrays.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

BASE_MODELS = ["lr", "knn", "nb", "rf", "et", "hgb", "xgb", "mlp"]
# Learners/implementations NOT in the simulated-committee pretraining pool (generalisation experiment)
ALT_MODELS = ["lgbm", "sgd", "bnb", "lsvc", "ridge", "bagdt"]


def make_model(name: str, seed: int):
    if name == "lr":
        return LogisticRegression(max_iter=2000)
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=15, weights="distance")
    if name == "nb":
        return GaussianNB()
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1)
    if name == "et":
        return ExtraTreesClassifier(n_estimators=300, random_state=seed, n_jobs=1)
    if name == "hgb":
        return HistGradientBoostingClassifier(random_state=seed)
    if name == "xgb":
        return XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6, n_jobs=1,
                             random_state=seed, verbosity=0, tree_method="hist")
    if name == "mlp":
        return MLPClassifier(hidden_layer_sizes=(128, 64), early_stopping=True, max_iter=300,
                             random_state=seed)
    if name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, n_jobs=1, random_state=seed,
                              verbose=-1)
    if name == "sgd":
        from sklearn.linear_model import SGDClassifier
        return SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=1000, random_state=seed)
    if name == "bnb":
        from sklearn.naive_bayes import BernoulliNB
        return BernoulliNB()
    if name == "lsvc":
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.svm import LinearSVC
        return CalibratedClassifierCV(LinearSVC(dual=False, max_iter=5000), cv=3)
    if name == "ridge":
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.linear_model import RidgeClassifier
        return CalibratedClassifierCV(RidgeClassifier(), cv=3, method="isotonic")
    if name == "bagdt":
        from sklearn.ensemble import BaggingClassifier
        return BaggingClassifier(DecisionTreeClassifier(), n_estimators=50, random_state=seed, n_jobs=1)
    raise ValueError(name)


def make_preprocessor(cat_mask: np.ndarray):
    num_idx = np.where(~cat_mask)[0]
    cat_idx = np.where(cat_mask)[0]
    parts = []
    if len(num_idx):
        parts.append(("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num_idx))
    if len(cat_idx):
        parts.append(("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                           OneHotEncoder(handle_unknown="ignore", max_categories=20,
                                                         sparse_output=False)), cat_idx))
    return ColumnTransformer(parts)


def _fit_predict(name, Xa, ya, Xb, n_classes, seed):
    """Fit model on (Xa, ya) and return probabilities on Xb aligned to all n_classes."""
    m = make_model(name, seed)
    # xgb needs contiguous labels 0..c-1 on the fitted subset
    present = np.unique(ya)
    remap = {c: i for i, c in enumerate(present)}
    ya_m = np.vectorize(remap.get)(ya) if name == "xgb" else ya
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(Xa, ya_m)
        p = m.predict_proba(Xb)
    out = np.zeros((Xb.shape[0], n_classes), dtype=np.float64)
    cls = present if name == "xgb" else m.classes_
    out[:, cls] = p
    out = np.clip(out, 0, None)
    s = out.sum(1, keepdims=True)
    out = np.where(s > 0, out / np.maximum(s, 1e-12), 1.0 / n_classes)
    return out.astype(np.float32)


def build_committee(X_tr, y_tr, X_te, n_classes, seed=42, inner_folds=5, models=BASE_MODELS):
    """Return P_oof (n_tr,K,C) and P_te (n_te,K,C). X_* are already preprocessed numeric arrays."""
    K = len(models)
    P_oof = np.zeros((len(y_tr), K, n_classes), dtype=np.float32)
    P_te = np.zeros((X_te.shape[0], K, n_classes), dtype=np.float32)
    min_count = np.bincount(y_tr, minlength=n_classes)
    n_inner = int(max(2, min(inner_folds, min_count[min_count > 0].min())))
    skf = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
    for k, name in enumerate(models):
        for a, b in skf.split(X_tr, y_tr):
            P_oof[b, k] = _fit_predict(name, X_tr[a], y_tr[a], X_tr[b], n_classes, seed)
        P_te[:, k] = _fit_predict(name, X_tr, y_tr, X_te, n_classes, seed)
    return P_oof, P_te


def save_split(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
