import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import pickle
import streamlit as st
from io import StringIO

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('Expresso_churn_dataset.csv')
    return df

def generate_comprehensive_report(df, max_rows=10000):
    """Generate a report on a sample of the data for better performance"""
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42)

    numeric_df = df.select_dtypes(include=[np.number])
    
    report = {
        "Dataset Overview": {
            "Shape": df.shape,
            "Columns": df.columns.tolist(),
            "Data Types": df.dtypes,
            "Missing Values": df.isnull().sum(),
            "Duplicate Rows": df.duplicated().sum()
        },
        "Descriptive Statistics": df.describe(),
        "Categorical Analysis": {
            col: {
                "Unique Values": df[col].nunique(),
                "Top Value": df[col].mode()[0],
                "Frequency (Top 5)": df[col].value_counts().head().to_dict()
            } for col in df.select_dtypes(include=['object']).columns
        },
        "Correlation Analysis": numeric_df.corr()
    }
    return report

def main():
    st.title("📊 Expresso Churn Prediction")
    
    df = load_data()
    
    # Data Exploration Section
    st.header("🔍 Data Exploration")
    
    if st.checkbox("Show raw data"):
        st.write(df.head())
    
    if st.checkbox("Show dataset info"):
        buffer = StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())
    
    if st.checkbox("Show missing values"):
        st.write(df.isnull().sum())
    
    if st.checkbox("Show duplicate rows"):
        st.write(f"Number of duplicates: {df.duplicated().sum()}")
    
    if st.button("📈 Generate Comprehensive Report"):
        with st.spinner('Generating report on a sample (10,000 rows)...'):
            report = generate_comprehensive_report(df)
            
            st.subheader("Dataset Overview")
            st.write(report["Dataset Overview"])
            
            st.subheader("Descriptive Statistics (sample)")
            st.write(report["Descriptive Statistics"].head(10))
            
            st.subheader("Categorical Analysis (Top 5)")
            for col, stats in report["Categorical Analysis"].items():
                with st.expander(f"Column: {col}"):
                    st.write(stats)
            
            st.subheader("Correlation Matrix (sample)")
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(report["Correlation Analysis"], annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            st.pyplot(fig)
    
    # Data Preprocessing Section
    st.header("⚙️ Data Preprocessing")
    
    if st.button("Preprocess Data"):
        with st.spinner('Preprocessing data...'):
            try:
                df_processed = df.drop(['user_id', 'REGION'], axis=1, errors='ignore')

                numeric_cols = df_processed.select_dtypes(include=np.number).columns
                categorical_cols = df_processed.select_dtypes(exclude=np.number).columns

                num_imputer = SimpleImputer(strategy='median')
                cat_imputer = SimpleImputer(strategy='most_frequent')
                df_processed[numeric_cols] = num_imputer.fit_transform(df_processed[numeric_cols])
                df_processed[categorical_cols] = cat_imputer.fit_transform(df_processed[categorical_cols])

                le = LabelEncoder()
                for col in categorical_cols:
                    df_processed[col] = le.fit_transform(df_processed[col].astype(str))

                st.session_state['df_processed'] = df_processed
                st.success("Preprocessing completed!")
                st.write("Processed Data Preview:")
                st.write(df_processed.head())
            except Exception as e:
                st.error(f"Preprocessing failed: {str(e)}")
    
    # Model Training Section
    st.header("🤖 Model Training")
    
    if st.button("Train Model"):
        if 'df_processed' not in st.session_state:
            st.warning("Please preprocess data first!")
            return
            
        df_processed = st.session_state['df_processed']
        
        with st.spinner('Training model...'):
            try:
                X = df_processed.drop('CHURN', axis=1)
                y = df_processed['CHURN']

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train_scaled, y_train)

                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)

                st.success(f"✅ Model trained successfully! Accuracy: {accuracy:.2f}")
                st.text("Classification Report:")
                st.text(classification_report(y_test, y_pred))

                with open('churn_model.pkl', 'wb') as f:
                    pickle.dump(model, f)
                with open('scaler.pkl', 'wb') as f:
                    pickle.dump(scaler, f)

                st.session_state['model_trained'] = True

            except Exception as e:
                st.error(f"Training failed: {str(e)}")
    
    # Prediction Section
    st.header("🔮 Churn Prediction")

    if st.button("Load Model"):
        try:
            with open('churn_model.pkl', 'rb') as f:
                st.session_state['model'] = pickle.load(f)
            with open('scaler.pkl', 'rb') as f:
                st.session_state['scaler'] = pickle.load(f)
            st.success("Model loaded successfully!")
        except FileNotFoundError:
            st.warning("No trained model found. Please train the model first.")
    
    if 'model' in st.session_state and 'scaler' in st.session_state:
        st.subheader("Enter Customer Details")

        input_data = {}
        feature_columns = [col for col in df.columns if col not in ['user_id', 'REGION', 'CHURN']]
        
        for col in feature_columns:
            if df[col].dtype == 'object':
                input_data[col] = st.selectbox(col, df[col].dropna().unique())
            else:
                input_data[col] = st.number_input(
                    col,
                    min_value=float(df[col].min()),
                    max_value=float(df[col].max()),
                    value=float(df[col].median())
                )

        if st.button("Predict Churn"):
            input_df = pd.DataFrame([input_data])

            numeric_cols = input_df.select_dtypes(include=np.number).columns
            categorical_cols = input_df.select_dtypes(exclude=np.number).columns

            input_df[numeric_cols] = SimpleImputer(strategy='median').fit_transform(input_df[numeric_cols])
            input_df[categorical_cols] = SimpleImputer(strategy='most_frequent').fit_transform(input_df[categorical_cols])

            le = LabelEncoder()
            for col in categorical_cols:
                input_df[col] = le.fit_transform(input_df[col].astype(str))

            input_scaled = st.session_state['scaler'].transform(input_df)
            prediction = st.session_state['model'].predict(input_scaled)
            prediction_proba = st.session_state['model'].predict_proba(input_scaled)

            st.subheader("Prediction Result")
            if prediction[0] == 1:
                st.error("⚠️ This customer is likely to churn.")
            else:
                st.success("✅ This customer is likely to stay.")

            st.write(f"🔢 Probability of churn: {prediction_proba[0][1]:.2f}")

if __name__ == "__main__":
    main()
