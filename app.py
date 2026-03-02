import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from utils.health_model import train_model
from utils.db import create_table, insert_record
from utils.medication import check_medication

# Setup
create_table()
model = train_model()

st.title("🏥 Healthcare Monitoring AI Agent")

menu = st.sidebar.radio("Navigation", ["Health Check", "Dashboard", "Medication"])

# ---------------- HEALTH CHECK ----------------#
if menu == "Health Check":

    st.subheader("Enter Health Details")

    heart_rate = st.number_input("Heart Rate", min_value=30, max_value=200)
    temperature = st.number_input("Temperature (°C)", min_value=30.0, max_value=45.0)
    bp = st.number_input("Blood Pressure", min_value=60, max_value=250)
    spo2 = st.number_input("SpO2 (%)", min_value=50, max_value=100)

    if st.button("Predict Risk"):

        input_data = [[heart_rate, temperature, bp, spo2]]

        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)

        risk_percent = round(probability[0][1] * 100, 2)

        if prediction[0] == 1:
            st.error("⚠ Health Risk Detected")
        else:
            st.success("✅ Normal Condition")

        st.write("### Risk Probability:", risk_percent, "%")

        insert_record(heart_rate, temperature, bp, spo2, int(prediction[0]))

# ---------------- DASHBOARD ----------------
elif menu == "Dashboard":
    st.subheader("📊 Health Data Dashboard")

    import sqlite3
    conn = sqlite3.connect("database.db")

    df = pd.read_sql_query("SELECT * FROM health_records", conn)

    if len(df) > 0:
        st.dataframe(df)

        st.line_chart(df[['heart_rate', 'bp']])

        risk_count = df['risk'].value_counts()

        st.bar_chart(risk_count)

        st.write("Normal Cases:", risk_count.get(0, 0))
        st.write("Risk Cases:", risk_count.get(1, 0))
    else:
        st.warning("No records yet.")

# ---------------- MEDICATION ----------------
elif menu == "Medication":
    st.subheader("Medication Reminder")

    med_name = st.text_input("Medicine Name")
    med_time = st.text_input("Time (HH:MM)")

    if st.button("Check Reminder"):
        message = check_medication(med_name, med_time)
        st.info(message)