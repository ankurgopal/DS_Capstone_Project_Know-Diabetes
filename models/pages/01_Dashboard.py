# models/pages/01_Dashboard.py
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)

import plotly.express as px
import plotly.graph_objects as go

import pickle
try:
    import joblib
except Exception:
    joblib = None

st.set_page_config(page_title="Model Dashboard", page_icon="📊", layout="centered")
st.title("📊 Model Dashboard")

# ---------- helpers ----------
def find_eval_csv() -> Path | None:
    here = Path(__file__).resolve()
    for p in [
        here.parent / "data" / "eval.csv",          # models/pages/data/eval.csv
        here.parent.parent / "data" / "eval.csv",   # models/data/eval.csv
        here.parents[2] / "data" / "eval.csv",      # repo_root/data/eval.csv
    ]:
        if p.exists():
            return p
    return None

def find_model_file() -> Path | None:
    models_dir = Path(__file__).resolve().parent.parent
    cands = list(models_dir.glob("*.pkl")) + list(models_dir.glob("*.joblib"))
    return cands[0] if cands else None

@st.cache_data(show_spinner=False)
def load_eval(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).dropna(subset=["y_true", "proba"])
    df["y_true"] = df["y_true"].astype(int)
    df["proba"]  = df["proba"].astype(float)
    return df

@st.cache_resource(show_spinner=False)
def load_model_name(path: Path | None) -> str:
    if path is None or not path.exists():
        return "Unknown model"
    try:
        mdl = None
        if joblib is not None:
            try: mdl = joblib.load(path)
            except Exception: pass
        if mdl is None:
            with open(path, "rb") as f: mdl = pickle.load(f)
        name = mdl.__class__.__name__
        if hasattr(mdl, "named_steps"):
            steps = list(mdl.named_steps.items())
            if steps:
                _, last_obj = steps[-1]
                name = f"{last_obj.__class__.__name__} in Pipeline"
        return name
    except Exception:
        return "Unknown model"

# ---------- load ----------
EVAL_PATH = find_eval_csv()
if EVAL_PATH is None:
    st.warning("Missing **data/eval.csv**. Export it from your notebook (true labels + probabilities).")
    st.stop()

df = load_eval(EVAL_PATH)
model_file = find_model_file()
model_name = load_model_name(model_file)
st.caption(f"Evaluating: **{model_name}** {f'({model_file.name})' if model_file else ''} • Data: `{EVAL_PATH.as_posix()}`")

default_thr = float(st.session_state.get("threshold", 0.50))
thr = st.slider("Decision threshold", 0.0, 1.0, default_thr, 0.01,
                help="If probability ≥ threshold ⇒ Positive.")

y_true = df["y_true"].values
proba  = df["proba"].values
y_pred = (proba >= thr).astype(int)

# ---------- KPIs ----------
tp = ((y_true==1)&(y_pred==1)).sum()
fn = ((y_true==1)&(y_pred==0)).sum()
fp = ((y_true==0)&(y_pred==1)).sum()
tn = ((y_true==0)&(y_pred==0)).sum()

pos_rate  = y_true.mean()
recall    = tp / (tp + fn + 1e-12)
precision = tp / (tp + fp + 1e-12)
accuracy  = (tp + tn) / (tp + tn + fp + fn + 1e-12)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Positive rate (dataset)", f"{pos_rate*100:.1f}%")
c2.metric("Recall (Sensitivity)",     f"{recall*100:.1f}%")
c3.metric("Precision (PPV)",          f"{precision*100:.1f}%")
c4.metric("Accuracy",                 f"{accuracy*100:.1f}%")

st.caption(
    "• **Positive rate (dataset)** = share of class-1 in this eval set. "
    "• **Recall** = of all true positives, how many we caught. "
    "• **Precision** = of all predicted positives, how many were correct. "
    "• **Accuracy** = overall correct. Raising the threshold ↑ typically **raises precision** but **lowers recall**."
)

# ---------- Confusion Matrix ----------
st.subheader("Confusion matrix")
cm = confusion_matrix(y_true, y_pred, labels=[0,1])
cm_df = pd.DataFrame(cm, index=["Actual 0","Actual 1"], columns=["Pred 0","Pred 1"])

fig_cm = go.Figure(data=go.Heatmap(
    z=cm_df.values, x=cm_df.columns, y=cm_df.index,
    colorscale="Blues", showscale=False, text=cm_df.values, texttemplate="%{text}"
))
fig_cm.update_layout(height=310, margin=dict(l=0, r=0, t=0, b=0))
st.plotly_chart(fig_cm, use_container_width=True)
st.dataframe(cm_df)
st.caption(
    "Confusion matrix at the **current threshold**. "
    "**TP**: predicted 1 & is 1, **FP**: predicted 1 but is 0, "
    "**FN**: predicted 0 but is 1, **TN**: predicted 0 & is 0."
)

# ---------- ROC / PR ----------
st.subheader("ROC / PR curves")
fpr, tpr, _ = roc_curve(y_true, proba)
roc_auc = auc(fpr, tpr)
prec, rec, _ = precision_recall_curve(y_true, proba)
ap = average_precision_score(y_true, proba)

fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC AUC = {roc_auc:.3f}"))
fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Chance", line=dict(dash="dash")))
fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=330)
st.plotly_chart(fig_roc, use_container_width=True)

fig_pr = go.Figure()
fig_pr.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=f"AP = {ap:.3f}"))
fig_pr.add_hline(y=pos_rate, line_dash="dash", annotation_text="Positive rate (dataset)")
fig_pr.update_layout(xaxis_title="Recall", yaxis_title="Precision", height=330)
st.plotly_chart(fig_pr, use_container_width=True)

st.caption(
    "**ROC** shows TPR vs FPR across thresholds (AUC≈0.5 is random; higher is better). "
    "**PR** shows Precision vs Recall; dashed line = dataset positive rate. "
    "PR/AP is more informative with class imbalance."
)

# ---------- Probability histogram ----------
st.subheader("Predicted probability distribution")
fig_hist = px.histogram(
    df, x="proba",
    color=df["y_true"].map({0:"Actual 0", 1:"Actual 1"}),
    nbins=30, barmode="overlay", opacity=0.6
)
fig_hist.add_vline(x=thr, line_dash="dot", annotation_text=f"thr={thr:.2f}")
fig_hist.update_layout(xaxis_title="Probability (class=1)", yaxis_title="Count", height=300)
st.plotly_chart(fig_hist, use_container_width=True)

st.caption(
    "Distribution of predicted probabilities by true class. "
    "The dotted line is the **decision threshold**. "
    "More separation ⇒ easier task; overlap near the threshold ⇒ likely errors."
)

# ---------- Cohort slices ----------
slice_cols = [c for c in ["BMI","Age","GenHlth","HighBP","HighChol","PhysActivity","Smoker"] if c in df.columns]
if slice_cols:
    st.subheader("Cohort slices")
    col = st.selectbox("Slice by", slice_cols)
    tmp = df.copy()

    def qcut_to_str_bins(series, q=5):
        cats = pd.qcut(series, q, duplicates="drop")
        ivals = list(cats.cat.categories)
        labels = [f"{iv.left:.1f}–{iv.right:.1f}" for iv in ivals]
        cats = cats.cat.rename_categories(labels)
        return cats.astype(str), labels

    order = None
    if pd.api.types.is_numeric_dtype(tmp[col]) and tmp[col].nunique() > 10:
        tmp[col], order = qcut_to_str_bins(tmp[col], q=5)
    elif pd.api.types.is_interval_dtype(tmp[col]):
        cats = tmp[col].astype("category")
        ivals = list(cats.cat.categories)
        labels = [f"{iv.left:.1f}–{iv.right:.1f}" for iv in ivals]
        tmp[col] = cats.cat.rename_categories(labels).astype(str)
        order = labels
    else:
        if pd.api.types.is_categorical_dtype(tmp[col]):
            order = list(tmp[col].cat.categories)

    grp = tmp.groupby(col, observed=True)["y_true"].mean().reset_index(name="positive_rate")

    fig_slice = px.bar(
        grp, x=col, y="positive_rate", text="positive_rate",
        labels={"positive_rate": "Positive rate (dataset)"},
        category_orders={col: order} if order else None
    )
    fig_slice.update_traces(texttemplate="%{text:.2f}")
    st.plotly_chart(fig_slice, use_container_width=True)

    st.caption(
        "Positive rate within each subgroup—useful to spot pockets of higher risk or data drift. "
        "Compare bars to the overall positive rate. Small groups can be **noisy**; descriptive, not causal."
    )
