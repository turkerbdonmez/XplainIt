# model_utils.py

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
)

def train_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = None
    if hasattr(model,"predict_proba"):
        y_prob = model.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    roc_ = None
    pr_ = None
    if y_prob is not None and len(np.unique(y_test))==2:
        roc_ = roc_auc_score(y_test, y_prob)
        pr_ = average_precision_score(y_test, y_prob)

    cm_ = confusion_matrix(y_test, y_pred)
    return {
        "model": model,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "roc_auc": roc_,
        "pr_auc": pr_,
        "cm": cm_
    }

def plot_and_save_confusion_matrix(cm,save_path,model_name="model"):
    fig,ax=plt.subplots(figsize=(4,4))
    disp=ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax,cmap="Blues",colorbar=False)
    ax.set_title(f"CM - {model_name}")
    fig.savefig(save_path,dpi=300,bbox_inches='tight')
    plt.close(fig)

def plot_and_save_roc_curve(y_test,y_prob,save_path,model_name="model"):
    fig,ax=plt.subplots()
    RocCurveDisplay.from_predictions(y_test,y_prob,ax=ax)
    ax.set_title(f"ROC - {model_name}")
    fig.savefig(save_path,dpi=300,bbox_inches='tight')
    plt.close(fig)

def plot_and_save_pr_curve(y_test,y_prob,save_path,model_name="model"):
    fig,ax=plt.subplots()
    PrecisionRecallDisplay.from_predictions(y_test,y_prob,ax=ax)
    ax.set_title(f"PR - {model_name}")
    fig.savefig(save_path,dpi=300,bbox_inches='tight')
    plt.close(fig)
