import streamlit as st
import numpy as np
import joblib

# Load model & scaler
model = joblib.load("model_rf.pkl")
scaler = joblib.load("scalerrf.pkl")

st.title("🫀 Diagnosa Penyakit Jantung")

st.write("Masukkan data pasien untuk prediksi:")

# Form input data pasien (HARUS sesuai urutan kolom dataset)
age = st.number_input("Usia", 20, 100, 50)
sex = st.selectbox("Jenis Kelamin (1=Laki-laki, 0=Perempuan)", [0, 1])
cp = st.selectbox("Tipe Nyeri Dada (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Tekanan Darah Istirahat", 80, 200, 120)
chol = st.number_input("Kolesterol", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1=Ya, 0=Tidak)", [0, 1])
restecg = st.selectbox("Hasil EKG Istirahat (0-2)", [0, 1, 2])
thalach = st.number_input("Detak Jantung Maksimum", 60, 250, 150)
exang = st.selectbox("Nyeri Dada Saat Olahraga (1=Ya, 0=Tidak)", [0, 1])
oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 10.0, 1.0)
slope = st.selectbox("Slope (0-2)", [0, 1, 2])
ca = st.selectbox("Jumlah Vessels (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (0=Normal, 1=Fixed Defect, 2=Reversible Defect)", [0, 1, 2])

# Buat array input
input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, 
                        thalach, exang, oldpeak, slope, ca, thal]])

# Scaling
input_scaled = scaler.transform(input_data)

# Prediksi
if st.button("Prediksi"):
    pred = model.predict(input_scaled)
    prob = model.predict_proba(input_scaled)

    if pred[0] == 1:
        st.error(f"⚠️ Pasien berpotensi memiliki penyakit jantung! (Prob: {prob[0][1]*100:.2f}%)")
    else:
        st.success(f"✅ Pasien tidak memiliki penyakit jantung. (Prob: {prob[0][0]*100:.2f}%)")

