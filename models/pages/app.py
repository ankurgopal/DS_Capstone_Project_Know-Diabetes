# app.py — short intro, tidy UI + recs + what-ifs + medical background + new "bad days" labels
import os, pickle, joblib, re, base64
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺", layout="centered")
st.title("🩺 Diabetes Risk Predictor")
st.caption("Enter your details to estimate diabetes screening risk. Adjust the left threshold to be stricter or more sensitive. Not medical advice.")

MODEL_FILE = "xgb_diabetes_model.pkl"   # change if needed
BACKGROUND_IMAGE_PATH = "assets/medical_bg.jpg"  # put a JPG/PNG here (or change the path)

# --------------------------- background styling (no image, always readable) ---------------------------
def apply_theme(theme: str = "Medical gradient"):
    if theme == "Medical gradient":
        app_bg = """
        .stApp {
            background-image:
              radial-gradient(1400px 500px at -10% -10%, #e9f4ff 0%, rgba(233,244,255,0) 60%),
              radial-gradient(1200px 600px at 110% -10%, #e6fff6 0%, rgba(230,255,246,0) 60%),
              linear-gradient(180deg, #ffffff, #f7fbff 60%, #ffffff 100%);
            background-attachment: fixed;
        }"""
    elif theme == "Calm teal":
        app_bg = """
        .stApp {
            background: linear-gradient(135deg, #e8fffb 0%, #ffffff 40%, #eff9ff 100%);
            background-attachment: fixed;
        }"""
    else:  # Midnight (dark backdrop with light card)
        app_bg = """
        .stApp { background: linear-gradient(180deg, #0f172a, #111827); }
        .stApp, .stApp h1, .stApp h2, .stApp h3, .stApp label, .stApp .stMarkdown { color: #e5e7eb; }
        """

    # Center content in a soft “card” so text is always readable
    card = """
    .block-container {
        max-width: 980px;
        margin: 0 auto;
        background-color: rgba(255,255,255,0.95);
        border-radius: 16px;
        padding: 1.25rem 2rem;
        box-shadow: 0 10px 30px rgba(2, 6, 23, 0.06);
    }
    /* Dark theme card */
    .stApp:has(.dark-card) .block-container { background-color: rgba(17,24,39,0.6); }
    """
    return f"<style>{app_bg}{card}</style>"


# --------------------------- load model ---------------------------
@st.cache_resource(show_spinner=False)
def _load_model(path: str, sig: float):
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)

model_path = None
for fn in [MODEL_FILE, "model.joblib", "model.pkl"]:
    if os.path.exists(fn):
        model_path = fn
        st.info(f"Found local **{fn}** — using that.")
        break

if not model_path:
    up = st.file_uploader("Upload trained Pipeline (.pkl/.joblib)", type=["pkl","joblib"])
    if up is not None:
        model_path = "uploaded_model.joblib"
        with open(model_path, "wb") as f: f.write(up.getbuffer())

if not model_path:
    st.warning("Place your model next to app.py or upload it above.")
    st.stop()

sig = os.path.getmtime(model_path)
model = _load_model(model_path, sig)
feat_names = list(getattr(model, "feature_names_in_", []))
if not feat_names:
    st.error("Model is missing `feature_names_in_`. Train on a pandas DataFrame or re-export.")
    st.stop()

def exists(col: str) -> bool: return col in feat_names
GEN_ONEHOTS = sorted([c for c in feat_names if re.fullmatch(r"General_Health_[2-5]", c)])
HAS_GEN_NUM = exists("GenHlth")

# --------------------------- helpers ---------------------------
HELP = {
    "PhysActivity":"Any physical activity in the past 30 days (outside of work).",
    "HvyAlcohol":"Men >14 drinks/week or Women >7 drinks/week (BRFSS).",
    "GenHlth":"1=Excellent, 2=Very good, 3=Good, 4=Fair, 5=Poor.",
    "Age":"BRFSS age categories: 1=18–24 … 13=80+.",
    "Income":"1=< $10k … 8=> $75k.",
    "Education":"1=None/Kindergarten … 6=College 4yrs+.",
    "Sex":"0=Female, 1=Male (dataset encoding).",
    "Threshold":"If the predicted probability ≥ threshold → predict class 1 (positive). "
                "Lowering increases recall; raising increases precision.",
}

def yesno(label: str, default: bool=False, help: str|None=None) -> bool:
    choice = st.radio(label, ["No", "Yes"], index=1 if default else 0, horizontal=True, help=help)
    return choice == "Yes"

def put(row: dict, col: str, val: float):
    if col in row: row[col] = float(val)

def render_grid(items, ncols=3):
    values = {}
    cols = st.columns(ncols)
    for i, (key, label, default, helpkey) in enumerate(items):
        with cols[i % ncols]:
            values[key] = yesno(label, default=default, help=HELP.get(helpkey))
    return values

def prob_band(p: float):
    if p < 0.15:   return ("Low", "Most people with similar answers screen negative.")
    if p < 0.30:   return ("Borderline", "Close to the cutoff; small changes could shift the result.")
    if p < 0.60:   return ("Elevated", "Higher likelihood of a positive screen — consider testing.")
    return ("High", "Strong likelihood of a positive screen — discuss formal testing.")

# --------------------------- sidebar (single threshold control @ 0.50) ---------------------------
with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Decision threshold (class 1)", 0.0, 1.0, 0.50, 0.01, help=HELP["Threshold"])

# --------------------------- form ---------------------------
st.subheader("Enter patient details")

r1c1, r1c2, r1c3 = st.columns(3)
with r1c1:
    sex_label = st.radio("Sex", ["Female","Male"], horizontal=True, help=HELP["Sex"])
    sex_val = 1.0 if sex_label == "Male" else 0.0
with r1c2:
    age_group = st.selectbox("Age group",
        ["18–24","25–29","30–34","35–39","40–44","45–49","50–54","55–59","60–64","65–69","70–74","75–79","80+"],
        index=6, help=HELP["Age"])
    AGE_CODE = {"18–24":1,"25–29":2,"30–34":3,"35–39":4,"40–44":5,"45–49":6,
                "50–54":7,"55–59":8,"60–64":9,"65–69":10,"70–74":11,"75–79":12,"80+":13}[age_group]
with r1c3:
    bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=27.0, step=0.1)

r2c1, r2c2, r2c3 = st.columns(3)
with r2c1:
    edu = st.selectbox("Education level",
        ["No school/Kindergarten","Grades 1–8","Grades 9–11","High school graduate",
         "Some college/tech","College 4yrs+"], index=5, help=HELP["Education"])
    EDU_CODE = {"No school/Kindergarten":1,"Grades 1–8":2,"Grades 9–11":3,
                "High school graduate":4,"Some college/tech":5,"College 4yrs+":6}[edu]
with r2c2:
    inc = st.selectbox("Household income",
        ["< $10k","$10–15k","$15–20k","$20–25k","$25–35k","$35–50k","$50–75k","> $75k"],
        index=7, help=HELP["Income"])
    INC_CODE = {"< $10k":1,"$10–15k":2,"$15–20k":3,"$20–25k":4,"$25–35k":5,"$35–50k":6,"$50–75k":7,"> $75k":8}[inc]
with r2c3:
    gh_opt = st.selectbox("General health (self-rated)",
                          ["Excellent","Very good","Good","Fair","Poor"],
                          index=2, help=HELP["GenHlth"])
    GH_CODE = {"Excellent":1,"Very good":2,"Good":3,"Fair":4,"Poor":5}[gh_opt]

# 🔁 Label change requested here:
d1, d2 = st.columns(2)
with d1:
    phys_days = st.slider("How many bad physical health days in the last 30 days?", 0, 30, 2)
with d2:
    ment_days = st.slider("How many bad mental health days in the last 30 days?", 0, 30, 2)

st.markdown("### Lifestyle")
lifestyle_items = [
    ("Smoker", "Current smoker?", False, "Smoker"),
    ("PhysActivity", "Regular physical activity?", True, "PhysActivity"),
    ("Fruits", "Eats fruit regularly?", True, "Fruits"),
    ("Veggies", "Eats vegetables regularly?", True, "Veggies"),
    ("HvyAlcoholConsump", "Heavy alcohol use?", False, "HvyAlcohol"),
    ("AnyHealthcare", "Has healthcare coverage?", True, "Sex"),
    ("NoDocbcCost", "Couldn't see doctor due to cost?", False, "Sex"),
    ("DiffWalk", "Difficulty walking?", False, "Sex"),
]
lifestyle_vals = render_grid(lifestyle_items, ncols=3)

st.markdown("### Medical history")
history_items = [
    ("HighBP", "Told you have high blood pressure?", False, "Sex"),
    ("HighChol", "Told you have high cholesterol?", False, "Sex"),
    ("CholCheck", "Cholesterol checked in past 5 years?", True, "Sex"),
    ("Stroke", "History of stroke?", False, "Sex"),
    ("HeartDiseaseorAttack", "Coronary heart disease (ever)?", False, "Sex"),
]
history_vals = render_grid(history_items, ncols=3)

predict = st.button("Predict")

# --------------------------- build model row ---------------------------
row = {c: 0.0 for c in feat_names}
def put_all():
    put(row, "Sex", sex_val)
    put(row, "Age", AGE_CODE); put(row, "Age_Group", AGE_CODE)
    put(row, "Education", EDU_CODE); put(row, "Education_Level", EDU_CODE)
    put(row, "Income", INC_CODE)
    put(row, "BMI", bmi); put(row, "PhysHlth", phys_days); put(row, "MentHlth", ment_days)
    if GEN_ONEHOTS:
        for c in GEN_ONEHOTS: row[c] = 0.0
        if GH_CODE in (2,3,4,5):
            put(row, f"General_Health_{GH_CODE}", 1.0)
    if HAS_GEN_NUM:
        put(row, "GenHlth", GH_CODE)
    aliases = {
        "Smoker": lifestyle_vals["Smoker"],
        "PhysActivity": lifestyle_vals["PhysActivity"],
        "Physical_Activity": lifestyle_vals["PhysActivity"],
        "Fruits": lifestyle_vals["Fruits"],
        "Veggies": lifestyle_vals["Veggies"],
        "HvyAlcoholConsump": lifestyle_vals["HvyAlcoholConsump"],
        "Heavy_Alcohol_Use": lifestyle_vals["HvyAlcoholConsump"],
        "AnyHealthcare": lifestyle_vals["AnyHealthcare"],
        "NoDocbcCost": lifestyle_vals["NoDocbcCost"],
        "Doctor_unavailable_cost": lifestyle_vals["NoDocbcCost"],
        "DiffWalk": lifestyle_vals["DiffWalk"],
        "Difficulty_Walking": lifestyle_vals["DiffWalk"],
        "HighBP": history_vals["HighBP"],
        "HighChol": history_vals["HighChol"],
        "CholCheck": history_vals["CholCheck"],
        "Stroke": history_vals["Stroke"],
        "HeartDiseaseorAttack": history_vals["HeartDiseaseorAttack"],
        "HeartDisease_or_Attack": history_vals["HeartDiseaseorAttack"],
    }
    for col, val in aliases.items():
        put(row, col, 1.0 if val else 0.0)

put_all()
X = pd.DataFrame([row], columns=feat_names)

# --------------------------- recommendation logic + what-ifs ---------------------------
def prob_band(p: float):
    if p < 0.15:   return ("Low", "Most people with similar answers screen negative.")
    if p < 0.30:   return ("Borderline", "Close to the cutoff; small changes could shift the result.")
    if p < 0.60:   return ("Elevated", "Higher likelihood of a positive screen — consider testing.")
    return ("High", "Strong likelihood of a positive screen — discuss formal testing.")

def make_recommendations(prob, pred, thr):
    recs = []
    if pred == 1:
        if prob >= 0.75:
            recs.append("Book an appointment with a healthcare provider soon to discuss formal testing (A1C / fasting plasma glucose / OGTT).")
        else:
            recs.append("Discuss diabetes screening with your healthcare provider (A1C / fasting plasma glucose / OGTT).")
        recs.append("This is a screening tool, not a diagnosis.")
    else:
        if prob >= (thr - 0.05):
            recs.append("Your result is close to the threshold. Consider re-checking after lifestyle changes or routine follow-up.")
        else:
            recs.append("Risk appears low right now. Maintain healthy habits and screen periodically.")
        recs.append("This is a screening tool, not a diagnosis.")
    if bmi >= 30: recs.append("Aim for a gradual 5–10% weight reduction over time; even modest loss can improve metabolic markers.")
    if not lifestyle_vals["PhysActivity"]: recs.append("Work toward ~150 minutes/week of moderate activity plus 2 days of strength training.")
    if not lifestyle_vals["Fruits"] or not lifestyle_vals["Veggies"]: recs.append("Increase fruit/vegetable and fiber intake; prioritize whole foods over refined sugars.")
    if lifestyle_vals["HvyAlcoholConsump"]: recs.append("Cut back on alcohol to within low-risk guidelines.")
    if lifestyle_vals["Smoker"]: recs.append("Consider a smoking cessation plan; smoking raises cardiometabolic risk.")
    if history_vals["HighBP"]: recs.append("Monitor and manage blood pressure with your provider.")
    if history_vals["HighChol"]: recs.append("Discuss cholesterol management and follow-up testing.")
    if history_vals["Stroke"] or history_vals["HeartDiseaseorAttack"]: recs.append("Given cardiovascular history, follow up promptly with your provider about glucose screening.")
    if lifestyle_vals["DiffWalk"]: recs.append("Ask about low-impact activity options or a physiotherapy plan to stay active safely.")
    if phys_days >= 14: recs.append("Consider a check-in about frequent ‘bad physical health days’.")
    if ment_days >= 14: recs.append("Consider mental health support if ‘bad mental health days’ are frequent.")
    return recs

def simulate_delta(X_base: pd.DataFrame, change: dict) -> float:
    if not hasattr(model, "predict_proba"):
        return np.nan
    X_new = X_base.copy()
    for k, v in change.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                if kk in X_new.columns:
                    X_new.loc[:, kk] = float(vv)
        else:
            if k in X_new.columns:
                X_new.loc[:, k] = float(v)
    return float(model.predict_proba(X_new)[0, 1])

def what_if_changes(X_base: pd.DataFrame) -> list[tuple[str, float]]:
    if not hasattr(model, "predict_proba"):
        return []
    base_p = float(model.predict_proba(X_base)[0, 1])
    candidates: list[tuple[str, dict]] = []
    if X_base.get("Smoker", pd.Series([0])).iloc[0] == 1:
        candidates.append(("Quit smoking", {"Smoker": 0}))
    if X_base.filter(regex="^Phys(ical_)?Activity$").max(axis=1).iloc[0] == 0:
        candidates.append(("Add regular physical activity", {"PhysActivity": 1, "Physical_Activity": 1}))
    if X_base.filter(regex="HvyAlcoholConsump|Heavy_Alcohol_Use").max(axis=1).iloc[0] == 1:
        candidates.append(("Reduce heavy alcohol use", {"HvyAlcoholConsump": 0, "Heavy_Alcohol_Use": 0}))
    if X_base.get("Fruits", pd.Series([1])).iloc[0] == 0:
        candidates.append(("Eat fruit regularly", {"Fruits": 1}))
    if X_base.get("Veggies", pd.Series([1])).iloc[0] == 0:
        candidates.append(("Eat vegetables regularly", {"Veggies": 1}))
    for drop, label in [(2, "Lower BMI by ~2"), (5, "Lower BMI by ~5")]:
        if "BMI" in X_base.columns:
            new_bmi = max(10.0, float(X_base["BMI"].iloc[0]) - drop)
            candidates.append((label, {"BMI": new_bmi}))
    impacts = []
    for label, change in candidates:
        new_p = simulate_delta(X_base, change)
        if not np.isnan(new_p):
            delta_pp = (base_p - new_p) * 100.0
            impacts.append((label, delta_pp))
    impacts.sort(key=lambda x: x[1], reverse=True)
    return impacts[:3]

# --------------------------- predict & display ---------------------------
if predict:
    try:
        if hasattr(model, "predict_proba"):
            p = float(model.predict_proba(X)[0, 1])
            y = int(p >= threshold)
        else:
            y_raw = model.predict(X)
            y = int(y_raw[0]) if hasattr(y_raw, "__len__") else int(y_raw)
            p = np.nan

        st.markdown("## Result")
        if y == 0:
            st.success(f"**Negative** 😀  (below threshold {threshold:.2f})")
        else:
            st.error(f"**Positive** 😟  (at/above threshold {threshold:.2f})")

        if not np.isnan(p):
            band, blurb = prob_band(p)
            as_100 = int(round(p * 100))
            st.metric("Estimated Probability (class=1)", f"{p:.3f}")
            st.info(
                f"**What this means:** About **{as_100} out of 100** people with similar answers "
                f"might screen **positive**. **Risk band:** *{band}*. {blurb}"
            )
        else:
            st.info("This model does not expose probabilities; showing class only.")

        prob_for_rec = 1.0 if np.isnan(p) and y == 1 else (0.0 if np.isnan(p) else p)
        recs = make_recommendations(prob_for_rec, y, threshold)
        st.markdown("## Recommendation")
        for r in recs:
            st.markdown(f"- {r}")

        if hasattr(model, "predict_proba"):
            st.markdown("## What-if: small changes")
            impacts = what_if_changes(X)
            if impacts:
                for label, dpp in impacts:
                    st.markdown(f"- **{label}** → estimated probability change **≈ −{dpp:.1f} pp**")
            else:
                st.caption("No obvious simple changes detected based on your answers.")

        st.caption("This app is for education/screening support only and is **not medical advice**. Please consult a licensed healthcare professional.")

        with st.expander("🔧 Debug: model input row"):
            st.dataframe(X)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
