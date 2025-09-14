# 🌍 AICTE Internship Project - Climate Risk and Disaster Management

This repository contains my AICTE Internship project on **Climate Risk and Disaster Management** using the **Global Natural Disasters Dataset** (derived from NASA’s EONET API via Kaggle). The project focuses on analyzing natural disasters such as wildfires, storms, and floods.

---

## 📌 Week 1 Project
In **Week 1**, the goal was to reach the **Data Understanding stage**:
- Imported necessary Python libraries (`pandas`, `numpy`)
- Loaded the dataset (`Data.csv`)
- Explored dataset using:
  - `.info()` → structure of data
  - `.describe()` → summary statistics
  - `.isnull().sum()` → missing values check
- Previewed disaster categories and their counts

This forms the foundation for deeper analysis and visualization in upcoming weeks.

---

## 📌 Week 2 Project
In **Week 2**, the project focused on **Exploratory Data Analysis (EDA), Data Transformation, and Feature Selection**:
- Utilized `matplotlib` and `seaborn` to visualize the distribution of disaster categories and the number of events over time.
- Handled missing values in the `Description` column by filling them with a placeholder.
- Split the `Geometry_Coordinates_1` column into `Longitude` and `Latitude` columns and created new features: `Month` and `DayOfWeek`.
- Dropped unnecessary columns to prepare a cleaned dataset (`processed_data.csv`) for the next phase.

---

## 📌 Week 3 Project
In **Week 3**, the focus was on **Machine Learning Model Development**:
- Defined features (`Longitude`, `Latitude`, `Month`, `DayOfWeek`) and the target variable (`Category_title`).
- Used `LabelEncoder` to transform the categorical target variable into a numerical format.
- Split the dataset into training and testing sets (80/20 ratio).
- Trained and evaluated multiple classification models, including `RandomForestClassifier`, `DecisionTreeClassifier`, and `LogisticRegression`.
- The primary Random Forest model achieved a **100% accuracy** on the test data.
- Saved the trained model (`disaster_rf_model.pkl`) and label encoder (`label_encoder.pkl`) for future use and deployment.

---

## 📌 Future Work
The upcoming phase will be:
- **Week 4** → Deployment of a Streamlit web application.

---

## 📂 Files in Repository
- `week1_project.ipynb` → Notebook for all week work.
- `app.py` → Streamlit web application script for deployment.
- `Data.csv` → The original Global Natural Disasters dataset.
- `processed_data.csv` → Cleaned and transformed dataset from Week 2.
- `disaster_rf_model.pkl` → The trained Random Forest model.
- `label_encoder.pkl` → The saved label encoder.
- `README.md` → Project description.

---

## ⚙️ How to Run
1. Clone the repository
   ```bash
   git clone [https://github.com/your-username/AICTE-ClimateRisk-Project.git](https://github.com/your-username/AICTE-ClimateRisk-Project.git)
