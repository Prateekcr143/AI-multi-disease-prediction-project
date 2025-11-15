#  AI Multi-Disease Prediction System  
# Predict **Heart, Kidney and Liver ** diseases using Machine Learning with a clean Streamlit UI.

---

##  Project Overview

This project is a **full-stack AI application** that predicts multiple diseases using trained ML models.  
It includes:

- ✔ Streamlit Web App  
- ✔ Trained ML Models (Random Forest Pipelines)  
- ✔ Data Preprocessing (Scaling, Encoding, Imputation)  
- ✔ Auto-generated Precautions  
- ✔ Clean UI with Tabs  
- ✔ Fully reproducible source code  

---

##  Supported Diseases

| Disease | File | Label Column |
|---------|-------|---------------|
| Heart Disease | `heart.csv` | `target` |
| Kidney Disease | `kidney.csv` | `classification` |
| Liver Disease | `liver.csv` | `Dataset` |

All datasets are preprocessed and used to train models under the `models/` folder.

---

##  Project Structure

AI-Multi-Disease-Prediction/
│── app.py # Streamlit UI
│── train_models.py # ML training script
│── models/
│ ├── heart_pipeline.pkl
│ ├── kidney_pipeline.pkl
│ ├── liver_pipeline.pkl
│ 
│── datasets/
│ ├── heart.csv
│ ├── kidney.csv
│ ├── liver.csv
│ 
│── README.md
│── requirements.txt

markdown
Copy code

---

##  Machine Learning

### **Model Used:**  
✔ Random Forest Classifier (200 trees)  

### **Preprocessing Pipeline Includes:**  
- Handling missing values (`SimpleImputer`)  
- Scaling numeric features (`StandardScaler`)  
- Encoding categorical values (`OneHotEncoder`)  
- Full ML Pipeline using `ColumnTransformer`

All models are saved as:

models/<disease>_pipeline.pkl


---

##  How to Run the Web App

### **1️ Install dependencies**

pip install -r requirements.txt



### **2️ Train models (optional)**

python train_models.py



### **3️ Run Streamlit App**

streamlit run app.py


---

##  Features of Streamlit App

### ✔ Multi-tab UI (Heart / Kidney / Liver)  
Each tab contains:

- ✦ Input fields  
- ✦ Automatic conversion of categorical inputs  
- ✦ Predictions using trained models  
- ✦ AI-generated precautions  
- ✦ Clean, responsive layout  

---

##  Tech Stack

| Component | Technology |
|-----------|-------------|
| Frontend | Streamlit |
| Backend | Python |
| ML Models | Scikit-learn |
| Data Storage | CSV Datasets |
| Deployment | Streamlit Cloud / Local Server |

---

##  Installation Requirements

streamlit
pandas
scikit-learn
joblib



---


---

##  Code Files Included

### ** app.py**
Contains complete Streamlit UI:

- Loads ML models
- Gets user inputs
- Generates predictions
- Shows disease-specific precautions

### ** .py**
Trains and exports:

- Heart model
- Kidney model
- Liver model
- Diabetes model

Outputs all `.pkl` files into the `models/` folder.

---


---



---

##  Author
**Prateek Kudari (Prateekcr143)**  

