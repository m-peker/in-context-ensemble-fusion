"""Fuse a small committee with a pretrained FusionPFN -- no meta-learner is trained.

Usage: python examples/quickstart.py path/to/fusionpfn.pt
"""
import sys

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "src")
from fusionpfn import fuse, load_model  # noqa: E402

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, train_size=128, stratify=y, random_state=0)
scaler = StandardScaler().fit(X_tr)
X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)

committee = {
    "logreg": LogisticRegression(max_iter=1000),
    "knn": KNeighborsClassifier(15),
    "naive_bayes": GaussianNB(),
    "random_forest": RandomForestClassifier(300, random_state=0),
}
# out-of-fold predictions on the training rows (context) and predictions of refitted models on the test rows
P_oof = np.stack([cross_val_predict(m, X_tr, y_tr, cv=5, method="predict_proba") for m in committee.values()], 1)
P_te = np.stack([m.fit(X_tr, y_tr).predict_proba(X_te) for m in committee.values()], 1)

model = load_model(sys.argv[1])
P, W = fuse(model, P_oof, y_tr, P_te, X_tr, X_te, n_perm=4, return_weights=True)

for k, name in enumerate(committee):
    print(f"{name:14s} log-loss {log_loss(y_te, P_te[:, k], labels=[0, 1]):.4f}  mean weight {W[:, k].mean():+.3f}")
print(f"{'average':14s} log-loss {log_loss(y_te, P_te.mean(1)):.4f}")
print(f"{'FusionPFN':14s} log-loss {log_loss(y_te, P):.4f}  accuracy {accuracy_score(y_te, P.argmax(1)):.4f}")
