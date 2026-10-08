# Diabetes Risk Screening App

A portfolio-ready **Python + Streamlit + Machine Learning** project.

## Features
- Clean dark clinical-style UI
- High-contrast white text
- Diabetes screening inputs
- ML model support
- Demo fallback if no model is present
- Responsive Streamlit layout

## Inputs
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Train the model

Place a `diabetes.csv` file in the project root with these columns:

```text
Pregnancies
Glucose
BloodPressure
SkinThickness
Insulin
BMI
DiabetesPedigreeFunction
Age
Outcome
```

Then run:

```bash
python train_model.py
```

The trained model will be saved to:

```text
model/diabetes_model.pkl
```

## Important
This project is for educational and portfolio use. It is not a medical diagnosis tool and should not be used for treatment decisions.
