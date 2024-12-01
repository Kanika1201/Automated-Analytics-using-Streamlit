import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import gdown

# Set in wide mode by default
st.set_page_config(page_title=None, page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)

# Title and description
st.title("Diabetes Prediction Results using Machine Learning Models")
st.write("Upload a file to get predictions from the pre-trained ML models or manually input values for prediction.")

# Model selection
# Model selection
model_choices = {
    "Logistic Regression": "https://drive.google.com/uc?id=1ymueuL6M1VsBa-eb-a5ICx1w30Q0ETyo",
    "KNN Model": "https://drive.google.com/uc?id=1BaHCNFRQzpoEZk4l7hKCGjypDrYhQcet",
    "Decision Tree": "https://drive.google.com/uc?id=1vi4BaIbM5gOxUGMpex3zuT53RM6Ed2OP",
    "Naive Bayes": "https://drive.google.com/uc?id=1BZoXDWwyzetF5lEkahzLTBN6SkO2CVaL",
    "SVM Model": "https://drive.google.com/uc?id=1PVqgz8t_eLu4mApkdlNkdCumdiGuKDPc",
    "Random Forest Model": "https://drive.google.com/uc?id=1HFYuJH49nks4iNrOJZCiLJF9KhIUCefJ"
}

# model_choices = {
#     "Logistic Regression": "lr_model.pkl",
#     "KNN Model": "knn_model.pkl",
#     "Decision Tree": "dt_classifier.pkl",
#     "Naive Bayes": "nb_model.pkl",
#     "SVM Model": "svm_model.pkl",
#     "Random Forest Model": "random_forest_model.pkl"
# }
selected_model = st.selectbox("Select a model to use for prediction:", list(model_choices.keys()))

# list of required columns
required_columns = [
    'gender', 'age', 'hypertension', 'heart_disease', 'smoking_encoded', 
    'bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes'
]

# Load the selected ML model
model_file = model_choices[selected_model]
with open(model_file, "rb") as file:
    model = pickle.load(file)

# Layout: Our display will be divided into two columns
col1, col2 = st.columns([1, 2])

# Place Manual input form in the left column (col1)
with col1:
    st.header("Manually Input Values for Prediction")

    # Creating a form for manual input
    with st.form(key='manual_input_form'):
        gender = st.selectbox('Gender', ['Male', 'Female'])
        age = st.number_input('Age', min_value=0, max_value=120, step=1)
        hypertension = st.selectbox('Hypertension (0 = No, 1 = Yes)', [0, 1])
        heart_disease = st.selectbox('Heart Disease (0 = No, 1 = Yes)', [0, 1])
        smoking_encoded = st.selectbox('Smoking (0 = No, 1 = Yes)', [0, 1])
        bmi = st.number_input('BMI', min_value=0.0, step=0.1)
        HbA1c_level = st.number_input('HbA1c Level', min_value=0.0, step=0.1)
        blood_glucose_level = st.number_input('Blood Glucose Level', min_value=0.0, step=0.1)

        # Submit button for form
        submit_button = st.form_submit_button("Predict")
    
    if submit_button:
        # In manual submission, map gender to numeric values
        gender_numeric = 1 if gender == 'Male' else 0  # 'Male' = 1, 'Female' = 0

        # Create a dictionary from the input values
        input_data = {
            'gender': gender_numeric,
            'age': age,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'smoking_encoded': smoking_encoded,
            'bmi': bmi,
            'HbA1c_level': HbA1c_level,
            'blood_glucose_level': blood_glucose_level
        }
        
        # Convert the input data into a DataFrame
        input_df = pd.DataFrame([input_data])

        # Make prediction based on the manual input
        prediction = model.predict(input_df)

        # Display the prediction result
        if prediction[0] == 1:
            # st.markdown("#### The model predicts a <span style='color:red;'>**POSITIVE DIABETES**</span> prediction.",unsafe_allow_html=True)
            st.markdown("<div style='backgroundColor:silver;padding:1.2rem;color:black;borderRadius:0.8rem;marginTop:2.4rem';'><h3>The model predicts a <span style='color:red;'><bold>POSITIVE DIABETES</bold></span> prediction.</h3></div>",unsafe_allow_html=True)
        else:
            # st.markdown("#### The model predicts a <span style='color:green;'>**NEGATIVE DIABETES**</span> prediction.",unsafe_allow_html=True)
            st.markdown("<div style='backgroundColor:silver;padding:1.2rem;color:black;borderRadius:0.8rem;marginTop:2.4rem';'><h3>The model predicts a <span style='color:green;'><bold>NEGATIVE DIABETES</bold></span> prediction.</h3></div>",unsafe_allow_html=True)


# Right column (col2) will contain the rest of things 
with col2:
    # Upload CSV File
    uploaded_file = st.file_uploader("Or choose a CSV file to make predictions for multiple entries:", type=["csv"])


    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.write("Uploaded Data:")
            st.write(data.head())

            # Check if required columns are in the uploaded file
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                st.warning(f"The following required columns are missing: {missing_columns}")
                for col in missing_columns:
                    data[col] = 0

            # Preprocessing button
            if st.button("Preprocess Data"):
                # Define preprocessing conditions
                conditions = (
                    (data['age'] > 0) &
                    (data['gender'].isin([0, 1])) &
                    (data['bmi'] > 10) &
                    (data['smoking_encoded'].isin([0, 1, 2, 3, 4])) &
                    (data['diabetes'].isin([0, 1]))
                )
                preprocessed_data = data[conditions].copy()
                rows_dropped = len(data) - len(preprocessed_data)
                st.write(f"Preprocessing complete. {rows_dropped} rows were dropped.")
                st.write("Preprocessed Data:")
                st.write(preprocessed_data)
                data = preprocessed_data

            # Predictions after preprocessing
            st.write(f"Running {selected_model} on the data...")
            predictions = model.predict(data[required_columns[:-1]])  # Exclude 'diabetes' (target column)
            data['Predictions'] = predictions
            st.write("Data with Predictions:")
            st.write(data)

            # Option to download preprocessed data
            csv = data.to_csv(index=False)
            st.download_button(
                label="Download Data with Predictions as CSV",
                data=csv,
                file_name="data_with_predictions.csv",
                mime="text/csv"
            )

            # Visualization: Input Data
            st.header("Input Data Visualizations")
            input_plot_choice = st.selectbox(
                "Select the plot you want to view for Input Data:",
                ["None", "Gender Distribution", "Smoking Status", "BMI Distribution", "Age Distribution", "Diabetes Status"]
            )

            if input_plot_choice == "Gender Distribution":
                # Map 0 to Female and 1 to Male
                data['gender'] = data['gender'].map({0: 'Female', 1: 'Male'})
                
                fig, ax = plt.subplots()
                sns.countplot(x='gender', data=data, ax=ax, palette='Set2')

                # Add value labels on top of each bar
                for p in ax.patches:
                    ax.text(
                        p.get_x() + p.get_width() / 2.,
                        p.get_height() + 0.1,            
                        int(p.get_height()),            
                        ha='center',                     # Horizontal alignment
                        va='bottom'                      # Vertical alignment
                    )
                
                ax.set_title('Gender Distribution')
                st.pyplot(fig)


            elif input_plot_choice == "Smoking Status":
                fig, ax = plt.subplots()
                sns.countplot(x='smoking_encoded', data=data, ax=ax, palette='Set1')

                # Add value labels on top of each bar
                for p in ax.patches:
                    ax.text(
                        p.get_x() + p.get_width() / 2.,
                        p.get_height() + 0.1,            
                        int(p.get_height()),            
                        ha='center',                     # Horizontal alignment
                        va='bottom'                      # Vertical alignment
                    )

                ax.set_title('Smoking Status')
                ax.set_xticklabels(['Never', 'Current', 'Former', 'Ever', 'Not Current'])
                st.pyplot(fig)


            elif input_plot_choice == "BMI Distribution":
                fig, ax = plt.subplots()
                sns.histplot(data['bmi'], kde=True, ax=ax, color='purple')
                ax.set_title('BMI Distribution')
                st.pyplot(fig)

            elif input_plot_choice == "Age Distribution":
                fig, ax = plt.subplots()
                sns.histplot(data['age'], kde=True, ax=ax, color='purple')
                ax.set_title('Age Distribution')
                st.pyplot(fig)

            elif input_plot_choice == "Diabetes Status":
                fig, ax = plt.subplots()
                data['diabetes'] = data['diabetes'].map({0: 'Non-Diabetic', 1: 'Diabetic'})
                sns.countplot(x='diabetes', data=data,  ax=ax, palette='Set2')

                # Add value labels on top of each bar
                for p in ax.patches:
                    ax.text(
                        p.get_x() + p.get_width() / 2.,
                        p.get_height() + 0.1,            
                        int(p.get_height()),            
                        ha='center',                     # Horizontal alignment
                        va='bottom'                      # Vertical alignment
                    )

                ax.set_title('Diabetes Status')
                st.pyplot(fig)

            # Visualization: Model Results
            st.header("Model Results Visualizations")
            result_plot_choice = st.selectbox(
                "Select the plot you want to view for Model Results:",
                ["None", "Confusion Matrix", "ROC Curve", "Prediction Distribution", "Feature Importance"]
            )

            if result_plot_choice == "Confusion Matrix" and 'diabetes' in data.columns:
                true_values = data['diabetes']
                cm = confusion_matrix(true_values, predictions)
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['True 0', 'True 1'])
                plt.title('Confusion Matrix')
                st.pyplot(fig)

            elif result_plot_choice == "ROC Curve" and 'diabetes' in data.columns:
                true_values = data['diabetes']
                fpr, tpr, _ = roc_curve(true_values, predictions)
                roc_auc = auc(fpr, tpr)
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.plot(fpr, tpr, color='b', label=f'ROC curve (AUC = {roc_auc:.2f})')
                ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
                ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate', title='ROC Curve')
                ax.legend(loc='lower right')
                st.pyplot(fig)

            elif result_plot_choice == "Prediction Distribution":
                fig, ax = plt.subplots(figsize=(6, 5))
                data['Predictions'] = data['Predictions'].map({0: 'Non-Diabetic', 1: 'Diabetic'})
                sns.countplot(x='Predictions', data=data, ax=ax, palette='Set2')
                
                # Add value labels on top of each bar
                for p in ax.patches:
                    ax.text(
                        p.get_x() + p.get_width() / 2.,
                        p.get_height() + 0.1,            
                        int(p.get_height()),            
                        ha='center',                     # Horizontal alignment
                        va='bottom'                      # Vertical alignment
                    )
                ax.set(title='Prediction Distribution', xlabel='Predicted Class', ylabel='Count')
                st.pyplot(fig)

            elif result_plot_choice == "Feature Importance" and isinstance(model, (RandomForestClassifier, DecisionTreeClassifier)):
                importances = model.feature_importances_
                feature_names = required_columns[:-1]  # Exclude 'diabetes' - target column
                indices = np.argsort(importances)[::-1]
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.barh(range(len(indices)), importances[indices], align="center")
                ax.set_yticks(range(len(indices)))
                ax.set_yticklabels([feature_names[i] for i in indices])
                ax.set_xlabel("Feature Importance")
                ax.set_title("Feature Importance")
                st.pyplot(fig)

        except Exception as e:
            st.error(f"An error occurred: {e}")
