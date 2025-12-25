import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="TA-11 Naive Bayes", layout="wide")

st.title("TA-12 — Naive Bayes (GaussianNB) | Customer Behaviour Prediction")
st.markdown(
    """
Aplikasi ini mengimplementasikan **Gaussian Naive Bayes** untuk memprediksi keputusan pembelian pelanggan (**Purchased**)  
berdasarkan **Gender, Age, EstimatedSalary** dari dataset *Customer Behaviour* (Kaggle).
"""
)

# ===================== DATASET =====================
st.header("1️⃣ Dataset")
st.caption("Pastikan file CSV ada di folder yang sama dengan app.py: `Customer_Behaviour.csv`")

try:
    df_raw = pd.read_csv("Customer_Behaviour.csv")
except Exception as e:
    st.error("CSV tidak kebaca. Pastikan ada file: Customer_Behaviour.csv")
    st.exception(e)
    st.stop()

st.write("Shape:", df_raw.shape)
st.dataframe(df_raw.head(10), use_container_width=True)

# ===================== PREPROCESS =====================
st.header("2️⃣ Preprocessing")

df = df_raw.copy()
if "User ID" in df.columns:
    df.drop("User ID", axis=1, inplace=True)

need_cols = {"Gender", "Age", "EstimatedSalary", "Purchased"}
if not need_cols.issubset(df.columns):
    st.error(f"Kolom CSV tidak sesuai. Harus ada: {need_cols}")
    st.write("Kolom yang ada:", list(df.columns))
    st.stop()

le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"].astype(str))

X = df.drop("Purchased", axis=1)
y = df["Purchased"].astype(int)

scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[["Age", "EstimatedSalary"]] = scaler.fit_transform(X[["Age", "EstimatedSalary"]])

st.success("✅ Encoding Gender + StandardScaler untuk Age & EstimatedSalary selesai.")

# ===================== MODEL CONFIG =====================
st.header("3️⃣ Model Configuration")

c1, c2 = st.columns(2)
test_size = c1.slider("Test size", 0.1, 0.4, 0.2, 0.05)
seed = c2.number_input("random_state", 0, 9999, 42, 1)

run = st.button("🚀 Jalankan Training & Evaluasi")

if run:
    # ===================== SPLIT =====================
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=float(test_size), random_state=int(seed), stratify=y
    )

    # ===================== TRAIN =====================
    model = GaussianNB()
    model.fit(X_train, y_train)

    # ===================== EVAL =====================
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["No Purchased=0", "Purchased=1"])

    st.subheader("📊 Hasil Evaluasi")
    st.metric("Accuracy", f"{acc:.4f}")

    a, b = st.columns(2)
    with a:
        st.write("Confusion Matrix")
        st.dataframe(
            pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]),
            use_container_width=True
        )
    with b:
        st.write("Classification Report")
        st.code(report, language="text")

    # ===================== predict_proba sample =====================
    st.subheader("📌 Predict Proba (Sample Test)")
    idx = 0
    sample = X_test.iloc[[idx]]
    st.write("True label:", int(y_test.iloc[idx]))
    st.write("Pred label:", int(model.predict(sample)[0]))
    st.write("Probabilitas [P(0), P(1)]:", model.predict_proba(sample)[0])

    # ===================== input user prediction =====================
    st.header("4️⃣ Prediksi Data Baru (Interaktif)")

    p1, p2, p3 = st.columns(3)
    gender_label = p1.selectbox("Gender", ["Female", "Male"])
    age_input = p2.slider("Age", 15, 60, 25)
    salary_input = p3.slider("EstimatedSalary", 10000, 200000, 70000, 1000)

    g = int(le.transform([gender_label])[0])
    new_raw = pd.DataFrame([[g, age_input, salary_input]], columns=X_scaled.columns)
    new_scaled = new_raw.copy()
    new_scaled[["Age", "EstimatedSalary"]] = scaler.transform(new_scaled[["Age", "EstimatedSalary"]])

    if st.button("🔮 Prediksi"):
        pred = int(model.predict(new_scaled)[0])
        proba = model.predict_proba(new_scaled)[0]
        st.success(f"Hasil Prediksi Class: {pred}")
        st.write("Probability [P(0), P(1)] :", proba)

    # ===================== dummy contrast =====================
    st.header("5️⃣ Simulasi 2 Data Dummy Kontras (TA-11)")

    dummy_raw = pd.DataFrame([
        {"Gender": g, "Age": 22, "EstimatedSalary": 120000},
        {"Gender": g, "Age": 55, "EstimatedSalary": 30000},
    ])

    dummy_scaled = dummy_raw.copy()
    dummy_scaled[["Age", "EstimatedSalary"]] = scaler.transform(dummy_scaled[["Age", "EstimatedSalary"]])

    dummy_pred = model.predict(dummy_scaled)
    dummy_proba = model.predict_proba(dummy_scaled)

    res = dummy_raw.copy()
    res["PredClass"] = dummy_pred
    res["P(0)"] = dummy_proba[:, 0]
    res["P(1)"] = dummy_proba[:, 1]

    st.dataframe(res, use_container_width=True)

    st.subheader("✅ Kesimpulan Singkat")
    st.write(
        "Model GaussianNB memprediksi keputusan pembelian berdasarkan fitur demografis dan finansial. "
        "Nilai probabilitas `predict_proba` dapat digunakan untuk melihat tingkat keyakinan model."
    )
else:
    st.info("Klik tombol **🚀 Jalankan Training & Evaluasi** untuk menampilkan hasil.")
