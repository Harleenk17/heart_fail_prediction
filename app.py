import gradio as gr
import joblib
import numpy as np

# Load trained model
model = joblib.load("model.pkl")

# Define prediction function
def predict_heart_failure(
    age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
    high_blood_pressure, platelets, serum_creatinine, serum_sodium,
    sex, smoking, time
):
    features = np.array([[age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
                          high_blood_pressure, platelets, serum_creatinine, serum_sodium,
                          sex, smoking, time]])
    prediction = model.predict(features)[0]
    return "DEATH EVENT" if prediction == 1 else "NO EVENT"

# Define inputs
inputs = [
    gr.Number(label="Age"),
    gr.Number(label="Anaemia (0 or 1)"),
    gr.Number(label="Creatinine Phosphokinase"),
    gr.Number(label="Diabetes (0 or 1)"),
    gr.Number(label="Ejection Fraction"),
    gr.Number(label="High Blood Pressure (0 or 1)"),
    gr.Number(label="Platelets"),
    gr.Number(label="Serum Creatinine"),
    gr.Number(label="Serum Sodium"),
    gr.Number(label="Sex (0 = female, 1 = male)"),
    gr.Number(label="Smoking (0 or 1)"),
    gr.Number(label="Follow-up Time (days)"),
]

gr.Interface(
    fn=predict_heart_failure,
    inputs=inputs,
    outputs="text",
    title="Heart Failure Prediction",
    description="Enter patient data to predict heart failure.",
).launch()
