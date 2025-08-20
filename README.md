# 🩺 Diabetes Prediction Project

## 📌 Overview
This project predicts the likelihood of diabetes in patients using machine learning models.  
The dataset used is the **Pima Indians Diabetes Database**, which contains medical diagnostic measurements such as glucose levels, blood pressure, BMI, insulin, etc.  

We applied **data preprocessing**, **model training**, and **evaluation** to compare two popular models:
- Gradient Boosting Classifier
- Random Forest Classifier

---

## 📂 Project Structure
```plaintext
├── data/
│   └── diabetes.csv              
├── src/
│   ├── __init__.py                
│   ├── preprocess.py               
│   ├── train.py                    
│   └── evaluate.py                 
├── main.py                         
├── requirements.txt                
└── README.md

## ⚙️ How It Works
1. **Preprocessing**  
   - Replace zero values in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` with NaN.  
   - Fill NaNs with median values.  
   - Split into features (X) and target (y).

2. **Training**  
   - Models used:
     - Gradient Boosting Classifier
     - Random Forest Classifier
   - Dataset split: 80% training, 20% testing.

3. **Evaluation**  
   - Accuracy and Classification Report generated.  
   - Custom threshold tuning improves performance.

## 📊 Model Performance

### Gradient Boosting Classifier (Best Threshold = 0.55)
- **Accuracy:** 93.51%  
- **Precision (Class 0):** 0.94  
- **Recall (Class 1):** 0.89  
- **F1-score (Class 1):** 0.91  

###Classification Report
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **0** | 0.94      | 0.96   | 0.95     | 100     |
| **1** | 0.92      | 0.89   | 0.91     | 54      |

**Overall Metrics**

| Metric          | Score |
|-----------------|-------|
| Accuracy        | 0.94  |
| Macro Avg F1    | 0.93  |
| Weighted Avg F1 | 0.93  | 

---

## ✅ Conclusion
- The **Gradient Boosting Classifier** provides the best performance with **93.5% accuracy**.  
- Random Forest also performs well (~91.6%), but GBC is the preferred model.  

---

## 🚀 How to Run

1️⃣ **Clone this repository**
```bash
git clone https://github.com/your-username/diabetes-prediction.git
cd diabetes-prediction
```

2️⃣ **Install dependencies**  
Make sure you have **Python 3.8+** installed. Then run:
```bash
pip install -r requirements.txt
```

3️⃣ **Run the main script**  
This will preprocess the data, train the models, and print evaluation results:
```bash
python main.py
```


