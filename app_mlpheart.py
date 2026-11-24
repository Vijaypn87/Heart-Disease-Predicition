"""Heart disease prediction app (Streamlit).

Trains SVM, KNN, Decision Tree and XGBoost on the Statlog heart dataset
and predicts the likelihood of heart disease for the values entered.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from streamlit_option_menu import option_menu
from xgboost import XGBClassifier

DATA_PATH = Path(__file__).parent / "G3_heart_disease.csv"
TARGET = "Heart Disease"
RANDOM_STATE = 45

# Coded values used in the dataset, with their clinical meaning.
CHEST_PAIN = {1: "Typical angina", 2: "Atypical angina", 3: "Non-anginal pain", 4: "Asymptomatic"}
EKG = {0: "Normal", 1: "ST-T wave abnormality", 2: "Left ventricular hypertrophy"}
SLOPE = {1: "Upsloping", 2: "Flat", 3: "Downsloping"}
THALLIUM = {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"}
YES_NO = {1: "Yes", 0: "No"}


def build_models():
    """SVM and KNN are distance based, so their features are standardised first."""
    return {
        "SVM": make_pipeline(StandardScaler(), SVC(probability=True, random_state=RANDOM_STATE)),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier()),
        "DECISION TREE": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "XGBOOST": XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss"),
    }


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data[TARGET] = data[TARGET].map({"Presence": 1, "Absence": 0})
    return data


@st.cache_resource
def train_models():
    """Train every model once per server process and score it on a held-out test set."""
    data = load_data()
    x = data.drop(columns=TARGET)
    y = data[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    trained, metrics = {}, {}
    for name, model in build_models().items():
        cv_scores = cross_val_score(model, x, y, cv=cv, scoring="accuracy")
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        trained[name] = model
        metrics[name] = {
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1 score": f1_score(y_test, pred, zero_division=0),
            "CV accuracy (5-fold)": cv_scores.mean(),
        }
    return trained, pd.DataFrame(metrics).T, list(x.columns)


def coded_select(label, mapping, **kwargs):
    """Selectbox that shows the meaning but returns the dataset's numeric code."""
    return st.selectbox(label, options=list(mapping), format_func=lambda k: mapping[k], **kwargs)


def user_inputs():
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=50, step=1)
        sex = st.selectbox("Sex", options=[1, 0], format_func=lambda k: "Male" if k else "Female")
        chest = coded_select("Chest pain type", CHEST_PAIN)
        bp = st.number_input("Resting blood pressure (mm Hg)", min_value=50, max_value=250, value=130, step=1)
        chol = st.number_input("Serum cholesterol (mg/dl)", min_value=100, max_value=600, value=240, step=1)
        fbs = coded_select("Fasting blood sugar over 120 mg/dl", YES_NO, index=1)
        ekg = coded_select("Resting EKG results", EKG)
    with col2:
        max_hr = st.number_input("Maximum heart rate achieved", min_value=60, max_value=220, value=150, step=1)
        angina = coded_select("Exercise-induced angina", YES_NO, index=1)
        st_dep = st.number_input("ST depression (exercise vs rest)", min_value=0.0, max_value=7.0, value=1.0, step=0.1)
        slope = coded_select("Slope of peak exercise ST segment", SLOPE)
        vessels = st.selectbox("Number of major vessels coloured by fluoroscopy", options=[0, 1, 2, 3])
        thallium = coded_select("Thallium stress test", THALLIUM)
    return [age, sex, chest, bp, chol, fbs, ekg, max_hr, angina, st_dep, slope, vessels, thallium]


def show_prediction(model, row, scores):
    pred = int(model.predict(row)[0])
    proba = model.predict_proba(row)[0]

    st.header("Diagnosis")
    if pred == 0:
        st.success(f"No heart disease predicted ({proba[0]:.0%} confidence)")
    else:
        st.error(f"Possibility of heart disease ({proba[1]:.0%} confidence)")

    fig, ax = plt.subplots()
    ax.pie(proba, labels=["No heart disease", "Heart disease"], colors=["#4caf50", "#e53935"],
           autopct="%1.1f%%", startangle=90)
    ax.set_title("Predicted probabilities")
    ax.axis("equal")
    st.pyplot(fig)
    plt.close(fig)

    st.header("Model performance (held-out test set)")
    cols = st.columns(len(scores))
    for col, (metric, value) in zip(cols, scores.items()):
        col.metric(metric, f"{value:.1%}")


def show_comparison(metrics):
    st.header("Model comparison")
    st.dataframe(metrics.style.format("{:.1%}"))

    for metric, color in [("Accuracy", "#e53935"), ("Precision", "#1e88e5"), ("CV accuracy (5-fold)", "#43a047")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        values = metrics[metric] * 100
        ax.bar(metrics.index, values, color=color)
        ax.set_xlabel("Model")
        ax.set_ylabel(f"{metric} (%)")
        ax.set_ylim(0, 100)
        ax.set_title(f"{metric} comparison")
        for i, v in enumerate(values):
            ax.text(i, v + 1, f"{v:.1f}", ha="center")
        st.pyplot(fig)
        plt.close(fig)

    best = metrics["CV accuracy (5-fold)"].idxmax()
    st.subheader(f"Best model by cross-validated accuracy: {best}")


def main():
    st.set_page_config(page_title="Heart Disease Prediction")
    st.title("Heart Disease Prediction")
    st.caption(
        "Educational demo only. This is not a medical device and must not be used "
        "for diagnosis or treatment decisions. Consult a qualified clinician."
    )

    models, metrics, feature_names = train_models()

    with st.sidebar:
        option = option_menu(
            "Select one of the following",
            options=["SVM", "KNN", "DECISION TREE", "XGBOOST", "MODEL GRAPHS"],
        )
        submit = st.button("Submit")

    row = pd.DataFrame([user_inputs()], columns=feature_names)

    if not submit:
        st.info("Enter the patient's values, pick a model in the sidebar and click Submit.")
    elif option == "MODEL GRAPHS":
        show_comparison(metrics)
    else:
        show_prediction(models[option], row, metrics.loc[option])


if __name__ == "__main__":
    main()
