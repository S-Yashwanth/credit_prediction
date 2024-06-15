import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import resample

@st.cache_data
def load_data(data, labels):
    data1 = pd.read_csv(data)  
    labels1 = pd.read_csv(labels)  
    return data1, labels1
scaler = StandardScaler()
st.title("Credit Risk Prediction")
labels_file = st.file_uploader("**Upload customer data**", type=["csv"])
data_file = st.file_uploader("**Upload payment details**", type=["csv"])

if data_file is not None and labels_file is not None:
    data, labels = load_data(data_file, labels_file)

    def preprocess_data(data, labels):
        numeric = data.select_dtypes(include=[np.number]).columns
        data[numeric] = data[numeric].fillna(data[numeric].mean())
        
        numeric_cols_labels = labels.select_dtypes(include=[np.number]).columns
        labels[numeric_cols_labels] = labels[numeric_cols_labels].fillna(labels[numeric_cols_labels].mean())

        merged_data = pd.merge(labels, data, left_on='id', right_on='id')
        st.write("Dataset")
        st.write(merged_data)

        date_columns = ['update_date', 'report_date']
        X = merged_data.drop(columns=['label', 'id'] + [col for col in date_columns if col in merged_data.columns])
        y = merged_data['label']
        

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        X_train_resampled, y_train_resampled = resample(
            X_train[y_train == 1],
            y_train[y_train == 1],
            replace=True,
            n_samples=X_train[y_train == 0].shape[0],
            random_state=42
        )
        X_train = pd.concat([X_train[y_train == 0], X_train_resampled])
        y_train = pd.concat([y_train[y_train == 0], y_train_resampled])

        
        X_test1 = X_test.copy()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
     

        return X_train, X_test, y_train, y_test, X_test1 

    X_train, X_test, y_train, y_test, X_test1 = preprocess_data(data, labels)


    def train_models(X_train, y_train):
        models = {
            'Logistic Regression': LogisticRegression(),
            'Random Forest': RandomForestClassifier(),
            'Gradient Boosting': GradientBoostingClassifier()
        }

        for name, model in models.items():
            model.fit(X_train, y_train)
            models[name] = model
        
        return models

    models = train_models(X_train, y_train)

    # Build
    st.title("Model")

    model_choice = st.selectbox("Choose a model", ['Logistic Regression', 'Gradient Boosting','Random Forest'])

    if model_choice:
        model = models[model_choice]
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Model: {model_choice}")
        st.write(f"Accuracy: {accuracy:.2f}")

   
    with st.sidebar.form("user_input_form"):
        st.title("User Input")
        id =    st.number_input("id", value=6789)
        fea_1 = st.number_input("fea_1", value=5)
        fea_2 = st.number_input("fea_2", value=1164.5)
        fea_3 = st.number_input("fea_3", value=3)
        fea_4 = st.number_input("fea_4", value=35000)
        fea_5 = st.number_input("fea_5", value=2)
        fea_6 = st.number_input("fea_6", value=15)
        fea_7 = st.number_input("fea_7", value=5)
        fea_8 = st.number_input("fea_8", value=67)
        fea_9 = st.number_input("fea_9", value=5)
        fea_10 = st.number_input("fea_10", value=60082)
        fea_11 = st.number_input("fea_11", value=200)
        OVD_t1 = st.number_input("OVD_t1", value=0)
        OVD_t2 = st.number_input("OVD_t2", value=0)
        OVD_t3 = st.number_input("OVD_t3", value=0)
        OVD_sum = st.number_input("OVD_sum", value=0)
        pay_normal = st.number_input("pay_normal", value=4)
        prod_code = st.number_input("prod_code", value=17)
        prod_limit = st.number_input("prod_limit", value=85789.7022)
        update_date = st.date_input("update_date")
        new_balance = st.number_input("new_balance", value=403903.2)
        highest_balance = st.number_input("highest_balance", value=440500)
        report_date = st.date_input("report_date")
        
        submit_button = st.form_submit_button("Predict")

    if submit_button:
        user_input = np.array([[fea_1, fea_2, fea_3, fea_4, fea_5, fea_6, fea_7, fea_8, fea_9, fea_10, fea_11,
                                OVD_t1, OVD_t2, OVD_t3, OVD_sum, pay_normal, prod_code, prod_limit, new_balance, highest_balance]])
        
        user_input_scaled = scaler.transform(user_input)
        
        prediction = model.predict(user_input_scaled)
        st.write("Credit Risk" if prediction[0] == 1 else "Credit Worthy")
else:
    st.write("Please upload both the data and labels files to proceed.")
