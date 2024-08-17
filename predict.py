import streamlit as st
import shap
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
shap.initjs()
from streamlit_shap import st_shap

df = pd.read_csv("dataset2.csv")

df.head()
df.shape
df.describe()
#NaN
print(f"\nNA values in dataset: \n{df.isna().sum()}")
print(f"\nPercentage NA values in dataset: \n{df.isna().sum()/len(df) * 100}")
X = df.drop('Disease', axis=1)
y = df['Disease']

# Initialize LabelEncoder
label_encoder = LabelEncoder()
# Convert any integer values in the dataset to strings
X_encoded = X.astype(str)
# Apply LabelEncoder to each column in the dataset
for column in X_encoded.columns:
    X_encoded[column] = label_encoder.fit_transform(X_encoded[column])

#split train & test
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, shuffle=True, test_size = 0.2)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# Model training
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# SHAP explainer
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)

# Streamlit App
st.title("Disease Prediction from Symptoms")

# Create user inputs for each symptom
user_input = {}
for symptom in X.columns:
    user_input[symptom] = st.checkbox(symptom)

# Convert user inputs into a DataFrame
input_data = pd.DataFrame([user_input])

# Predict the disease
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.write(f"Based on the symptoms, you might have: **{prediction[0]}**")

# SHAP explanation for the input
shap_values_input = explainer.shap_values(input_df)


# Force plot
st.subheader("Force Plot")
# fig, ax = plt.subplots()
# shap.plots.force(explainer.expected_value[0], shap_values_input[0,:], input_df.iloc[0,:], matplotlib=True)
st_shap(shap.force_plot(explainer.expected_value[0], shap_values_input[0], input_df), height=400, width=1000)

# st.write(input_df)
# st.pyplot(fig,bbox_inches='tight')

# Decision plot
st.subheader("Decision Plot")
# fig, ax = plt.subplots()
# shap.decision_plot(explainer.expected_value[0], shap_values_input[0], X_test.columns)
st_shap(shap.decision_plot(explainer.expected_value[0], shap_values_input[0], X_test.columns))
# st.pyplot(fig)

