import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# Load data
@st.cache
def load_data():
    return pd.read_csv('dataset.csv')

df = load_data()

# Convert categorical variables
df_encoded = pd.get_dummies(df, columns=['sales', 'salary'])

# Define features and target
X = df_encoded.drop('left', axis=1)
y = df_encoded['left']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train models
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

model_logistic = LogisticRegression(max_iter=1000)
model_logistic.fit(X_train, y_train)

# Streamlit App
st.title("Employee Turnover Prediction")

st.sidebar.title("Options")
model_option = st.sidebar.selectbox("Select Model", ["Random Forest", "Logistic Regression"])

# Input features
st.sidebar.header("Input Features")

def user_input_features():
    satisfaction_level = st.sidebar.slider('Satisfaction Level', 0.0, 1.0, 0.5)
    last_evaluation = st.sidebar.slider('Last Evaluation', 0.0, 1.0, 0.5)
    number_project = st.sidebar.slider('Number of Projects', 2, 7, 3)
    average_monthly_hours = st.sidebar.slider('Average Monthly Hours', 96, 310, 200)
    time_spend_company = st.sidebar.slider('Time Spent at Company', 2, 10, 5)
    work_accident = st.sidebar.selectbox('Work Accident', [0, 1])
    promotion_last_5years = st.sidebar.selectbox('Promotion Last 5 Years', [0, 1])
    department = st.sidebar.selectbox('Department', ['sales', 'accounting', 'hr', 'management', 'marketing', 'product_mng', 'sales', 'support', 'technical'])
    salary = st.sidebar.selectbox('Salary', ['low', 'medium', 'high'])
    
    input_data = {
        'satisfaction_level': satisfaction_level,
        'last_evaluation': last_evaluation,
        'number_project': number_project,
        'average_montly_hours': average_monthly_hours,
        'time_spend_company': time_spend_company,
        'Work_accident': work_accident,
        'promotion_last_5years': promotion_last_5years,
        'sales': department,
        'salary': salary
    }
    
    return pd.DataFrame(input_data, index=[0])

input_df = user_input_features()

# Convert user input to model-compatible format
input_df_encoded = pd.get_dummies(input_df, columns=['sales', 'salary'])
input_df_encoded = input_df_encoded.reindex(columns=X.columns, fill_value=0)

# Predict
if st.sidebar.button('Predict'):
    if model_option == "Random Forest":
        prediction = model_rf.predict(input_df_encoded)
        prediction_proba = model_rf.predict_proba(input_df_encoded)
    else:
        prediction = model_logistic.predict(input_df_encoded)
        prediction_proba = model_logistic.predict_proba(input_df_encoded)
    
    st.write("### Prediction")
    st.write("Employee will leave the company: " + ("Yes" if prediction[0] == 1 else "No"))
    
    st.write("### Prediction Probability")
    st.write("Probability of leaving: {:.2f}".format(prediction_proba[0][1]))
    st.write("Probability of staying: {:.2f}".format(prediction_proba[0][0]))
    
    # ROC Curve
    st.write("### ROC Curve")
    if model_option == "Random Forest":
        roc_auc = roc_auc_score(y_test, model_rf.predict_proba(X_test)[:, 1])
        fpr, tpr, _ = roc_curve(y_test, model_rf.predict_proba(X_test)[:, 1])
    else:
        roc_auc = roc_auc_score(y_test, model_logistic.predict_proba(X_test)[:, 1])
        fpr, tpr, _ = roc_curve(y_test, model_logistic.predict_proba(X_test)[:, 1])
    
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    st.pyplot()
