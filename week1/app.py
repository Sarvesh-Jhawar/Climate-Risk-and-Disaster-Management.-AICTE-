import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# ------------------------------
# Load and Preprocess Data
# ------------------------------
data = pd.read_csv("processed_data.csv")

# Drop columns we don’t need
if "Datetime" in data.columns:
    data = data.drop(columns=["Datetime"])

# Encode target only (Category_title)
le = LabelEncoder()
data["Category_title"] = le.fit_transform(data["Category_title"])

# Features & Target
X = data.drop("Category_title", axis=1)
y = data["Category_title"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Climate Risk & Disaster Management", layout="wide")

st.title("🌍 Climate Risk & Disaster Management - AI Model")
st.markdown("This app predicts disaster categories using multiple machine learning models. 🚀")

# Sidebar
st.sidebar.header("⚙️ Choose Model")
classifier_name = st.sidebar.selectbox(
    "Select Classifier",
    ("Decision Tree", "Random Forest", "Logistic Regression", "KNN", "Naive Bayes", "SVM")
)

st.sidebar.header("📥 Enter Disaster Event Details")
longitude = st.sidebar.number_input("Longitude", min_value=-180.0, max_value=180.0, value=0.0)
latitude = st.sidebar.number_input("Latitude", min_value=-90.0, max_value=90.0, value=0.0)
month = st.sidebar.number_input("Month (1-12)", min_value=1, max_value=12, value=1)
dayofweek = st.sidebar.number_input("Day of Week (0=Mon, 6=Sun)", min_value=0, max_value=6, value=0)

input_data = [[longitude, latitude, month, dayofweek]]

# ------------------------------
# Function to get classifier
# ------------------------------
def get_classifier(clf_name):
    if clf_name == "Decision Tree":
        return DecisionTreeClassifier()
    elif clf_name == "Random Forest":
        return RandomForestClassifier()
    elif clf_name == "Logistic Regression":
        return LogisticRegression(max_iter=200)
    elif clf_name == "KNN":
        return KNeighborsClassifier()
    elif clf_name == "Naive Bayes":
        return GaussianNB()
    elif clf_name == "SVM":
        return SVC(probability=True)

# ------------------------------
# Training & Prediction
# ------------------------------
if st.sidebar.button("🔮 Predict Disaster Category"):
    with st.spinner("Training model... Please wait ⏳"):
        clf = get_classifier(classifier_name)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        pred = clf.predict(input_data)[0]
        pred_label = le.inverse_transform([pred])[0]

    # Results layout
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"✅ Accuracy on Test Data: **{round(acc,3)}**")
        st.info(f"🔮 Predicted Disaster Category: **{pred_label}**")

        # Confidence Scores
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(input_data)[0]
            proba_df = pd.DataFrame({
                "Category": le.inverse_transform(range(len(proba))),
                "Probability": proba
            })
            st.write("### 🎯 Prediction Confidence")
            st.bar_chart(proba_df.set_index("Category"))

    with col2:
        st.write("### 📈 Classification Report")
        st.text(classification_report(y_test, y_pred))

    # Confusion Matrix Heatmap
    st.write("### 🧩 Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    # Feature Importance for Tree Models
    if classifier_name in ["Decision Tree", "Random Forest"]:
        st.write("### 🌟 Feature Importance")
        importance = clf.feature_importances_
        feat_df = pd.DataFrame({"Feature": X.columns, "Importance": importance})
        feat_df = feat_df.sort_values("Importance", ascending=False)
        st.bar_chart(feat_df.set_index("Feature"))

    # Map Visualization
    st.write("### 🗺️ Disaster Event Location")
    map_df = pd.DataFrame({"lat": [latitude], "lon": [longitude]})
    st.map(map_df)

    # Downloadable Report
    buffer = io.StringIO()
    buffer.write("Model: {}\n".format(classifier_name))
    buffer.write("Accuracy: {}\n".format(round(acc, 3)))
    buffer.write("Prediction: {}\n".format(pred_label))
    st.download_button("⬇️ Download Results", buffer.getvalue(), "results.txt")

# ------------------------------
# Dataset Explorer
# ------------------------------
st.sidebar.markdown("### 📊 Dataset Overview")
st.sidebar.write("Number of records:", data.shape[0])
st.sidebar.write("Number of features:", data.shape[1] - 1)
st.sidebar.bar_chart(data["Category_title"].value_counts())

if st.checkbox("👀 Show Sample Data"):
    st.dataframe(data.head(10))
