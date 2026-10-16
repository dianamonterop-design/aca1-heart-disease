"""App Streamlit: predicción de enfermedad cardíaca (UCI Heart Disease).

Uso:  streamlit run app.py   (requiere haber ejecutado antes: python train.py)
"""
import json

import joblib
import pandas as pd
import streamlit as st

from heart_ml import FEATURES, METRICS_PATH, MODEL_PATH, load_data

st.set_page_config(page_title="Predicción de enfermedad cardíaca", page_icon="❤️", layout="wide")


@st.cache_resource
def get_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def get_data():
    return load_data()


@st.cache_data
def get_metrics():
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


if not MODEL_PATH.exists():
    st.error("No se encontró model.joblib. Ejecuta primero `python train.py`.")
    st.stop()

model, df, metrics = get_model(), get_data(), get_metrics()

st.title("❤️ Detección asistida de enfermedad cardíaca")
st.caption(
    "Proyecto ACA N° 1 · Aprendizaje Automático · CUN · "
    "Modelo clásico de ML sobre el dataset UCI Heart Disease. "
    "Herramienta académica: no sustituye un diagnóstico médico."
)

tab_pred, tab_model, tab_data = st.tabs(["🔮 Predicción", "📊 Modelo y métricas", "🗂️ Datos"])

# ---------------------------------------------------------------- Predicción
with tab_pred:
    st.subheader("Ingresa los datos del paciente")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider("Edad", 20, 90, 55)
        sex = st.radio("Sexo", [1, 0], format_func=lambda v: "Masculino" if v else "Femenino", horizontal=True)
        cp = st.selectbox(
            "Tipo de dolor de pecho",
            [1, 2, 3, 4],
            format_func=lambda v: {1: "Angina típica", 2: "Angina atípica", 3: "Dolor no anginoso", 4: "Asintomático"}[v],
        )
        trestbps = st.slider("Presión arterial en reposo (mm Hg)", 80, 220, 130)
    with c2:
        chol = st.slider("Colesterol sérico (mg/dl)", 100, 600, 240)
        fbs = st.radio("Glucemia en ayunas > 120 mg/dl", [0, 1], format_func=lambda v: "Sí" if v else "No", horizontal=True)
        restecg = st.selectbox(
            "ECG en reposo",
            [0, 1, 2],
            format_func=lambda v: {0: "Normal", 1: "Anomalía onda ST-T", 2: "Hipertrofia ventricular izq."}[v],
        )
        thalach = st.slider("Frecuencia cardíaca máxima", 60, 220, 150)
    with c3:
        exang = st.radio("Angina inducida por ejercicio", [0, 1], format_func=lambda v: "Sí" if v else "No", horizontal=True)
        oldpeak = st.slider("Depresión del ST (oldpeak)", 0.0, 7.0, 1.0, 0.1)
        slope = st.selectbox(
            "Pendiente del segmento ST",
            [1, 2, 3],
            format_func=lambda v: {1: "Ascendente", 2: "Plana", 3: "Descendente"}[v],
        )
        ca = st.selectbox("Vasos mayores coloreados (0-3)", [0, 1, 2, 3])
        thal = st.selectbox(
            "Talasemia",
            [3, 6, 7],
            format_func=lambda v: {3: "Normal", 6: "Defecto fijo", 7: "Defecto reversible"}[v],
        )

    if st.button("Predecir", type="primary"):
        row = pd.DataFrame(
            [{
                "age": age, "trestbps": trestbps, "chol": chol, "thalach": thalach,
                "oldpeak": oldpeak, "sex": sex, "cp": cp, "fbs": fbs, "restecg": restecg,
                "exang": exang, "slope": slope, "ca": float(ca), "thal": float(thal),
            }]
        )[FEATURES]
        proba = float(model.predict_proba(row)[0, 1])
        st.metric("Probabilidad estimada de enfermedad cardíaca", f"{proba:.1%}")
        st.progress(proba)
        if proba >= 0.5:
            st.error("El modelo clasifica este caso como **riesgo alto** de enfermedad cardíaca.")
        else:
            st.success("El modelo clasifica este caso como **riesgo bajo** de enfermedad cardíaca.")

# ------------------------------------------------------------------- Métricas
with tab_model:
    best = metrics["best_model"]
    st.subheader(f"Modelo seleccionado: {best}")
    res = metrics["results"]
    table = pd.DataFrame(res).T.drop(columns=["best_params", "confusion_matrix"])
    st.dataframe(table, use_container_width=True)

    m = res[best]
    cols = st.columns(5)
    for col, key in zip(cols, ["accuracy", "precision", "recall", "f1", "roc_auc"]):
        col.metric(key.upper(), f"{m[key]:.3f}")

    st.write("**Matriz de confusión** (conjunto de prueba)")
    st.dataframe(
        pd.DataFrame(
            m["confusion_matrix"],
            index=["Real: sano", "Real: enfermo"],
            columns=["Pred: sano", "Pred: enfermo"],
        )
    )
    st.caption("En un contexto clínico es clave el recall: un falso negativo es más costoso que un falso positivo.")

# ---------------------------------------------------------------------- Datos
with tab_data:
    st.subheader("Dataset UCI Heart Disease")
    st.write(f"{df.shape[0]} registros · {df.shape[1]} columnas · {df['target'].mean():.1%} con enfermedad")
    st.dataframe(df.head(50), use_container_width=True)
    var = st.selectbox("Distribución de una variable numérica", ["age", "trestbps", "chol", "thalach", "oldpeak"])
    st.bar_chart(df[var].value_counts(bins=15).sort_index().rename(var).reset_index(drop=True))
