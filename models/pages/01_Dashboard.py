# models/pages/01_Dashboard.py
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score

# Plotly for nice charts
try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    st.error("Plotly is required. Add `plotly` to requirements.txt and redeploy.")
    st.stop()

st.set_page_config(page_title="Model Dashboard", page_icon="📊", layout="centered")
st.title("📊 Model Dashboard")

def find_eval_csv() -> Path | None:
    """Look for data/eval.csv in common locations relative to this file."""
    here = Path(__file__).resolve()           # .../models/pages/01_Dashboard.py
    candidates = [
        here.parent / "data" / "eval.csv",            # models/pages/data/eval.csv
        here.parent.parent / "data" / "eval.csv",     # models/data/eval.csv
        here.parents[2] / "data" / "eval.csv",        # repo_root/data/eval.csv
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

DATA_PATH = find_eval_csv()
if DATA_PATH is None:
    st.warning("Missing **data/eval.csv**. Export it from your notebook (true labels + probabilities).")
    st.stop()

@st.cache_data(show_spinner=False)
def load_eval(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["y_true", "proba"])
    df["y_true"] = df["y_true"].astype(int)
    df["proba"]  = df["proba"].astype(float)
    return df

df = load_eval(DATA_PATH)

# If your main page stored threshold in session_state, reuse it
thr_default = float(st.session_state.get("threshold", 0.50))
thr = st.slider("Decision threshold", 0.0, 1.0, thr_default, 0.01,
                help="If probability ≥ threshold ⇒ Positive.")

y_true = df["y_true"].values
proba  = df["proba"].values
y_pred = (proba >= thr).astype(int)

# KPIs
tp = ((y_true==1)&(y_pred==1)).sum()
fn = ((y_true==1)&(y_pred==0)).sum()
fp = ((y_true==0)&(y_pred==1)).sum()
tn = ((y_true==0)&(y_pred==0)).sum()

prevalence = y_true.mean()
recall     = tp / (tp + fn + 1e-12)
precision  = tp / (tp + fp + 1e-12)
accuracy   = (tp + tn) / (tp + tn + fp + fn + 1e-12)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Prevalence", f"{prevalence*100:.1f}%")
c2.metric("Recall (Sens.)", f"{recall*100:.1f}%")
c3.metric("Precision (PPV)", f"{precision*100:.1f}%")
c4.metric("Accuracy", f"{accuracy*100:.1f}%")

# Confusion matrix
st.subheader("Confusion matrix")
cm = confusion_matrix(y_true, y_pred, labels=[0,1])
st.write(pd.DataFrame(cm, index=["Actual 0","Actual 1"], columns=["Pred 0","Pred 1"]))

# ROC curve
st.subheader("ROC / PR curves")
fpr, tpr, _ = roc_curve(y_true, proba)
roc_auc = auc(fpr, tpr)
prec, rec, _ = precision_recall_curve(y_true, proba)
ap = average_precision_score(y_true, proba)

fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC AUC = {roc_auc:.3f}"))
fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Chance", line=dict(dash="dash")))
fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=340)
st.plotly_chart(fig_roc, use_container_width=True)

# PR curve
fig_pr = go.Figure()
fig_pr.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=f"AP = {ap:.3f}"))
fig_pr.add_hline(y=prevalence, line_dash="dash", annotation_text="Prevalence")
fig_pr.update_layout(xaxis_title="Recall", yaxis_title="Precision", height=340)
st.plotly_chart(fig_pr, use_container_width=True)

# Probability histogram
st.subheader("Predicted probability distribution")
fig_hist = px.histogram(df, x="proba", color=df["y_true"].map({0:"Actual 0", 1:"Actual 1"}),
                        nbins=30, barmode="overlay", opacity=0.6)
fig_hist.add_vline(x=thr, line_dash="dot", annotation_text=f"thr={thr:.2f}")
fig_hist.update_layout(xaxis_title="Probability (class=1)", yaxis_title="Count", height=300)
st.plotly_chart(fig_hist, use_container_width=True)

# Optional cohort slices if present
slice_cols = [c for c in ["BMI","Age","GenHlth","HighBP","HighChol","PhysActivity","Smoker"] if c in df.columns]
if slice_cols:
    st.subheader("Cohort slices")
    col = st.selectbox("Slice by", slice_cols)
    tmp = df.copy()
    if pd.api.types.is_numeric_dtype(tmp[col]) and tmp[col].nunique() > 10:
        tmp[col] = pd.qcut(tmp[col], 5, duplicates="drop")
    grp = tmp.groupby(col)["y_true"].mean().reset_index(name="prevalence")
    fig_slice = px.bar(grp, x=col, y="prevalence", text="prevalence",
                       labels={"prevalence":"Prevalence"})
    fig_slice.update_traces(texttemplate="%{text:.2f}")
    st.plotly_chart(fig_slice, use_container_width=True)

