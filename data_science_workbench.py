# Enhanced Data Science Workbench with AutoML and Real-Time Charting
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.pipeline import make_pipeline
from statsmodels.tsa.arima.model import ARIMA
from lazypredict.Supervised import LazyClassifier, LazyRegressor
import joblib
import io
import time
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Data Science Workbench", layout="wide")
st.title("📊 Data Science Workbench")

menu = st.sidebar.radio("Navigation", [
    "Home", "EDA", "Modeling", "AutoML", "Prediction", "Unsupervised Learning", "Model Evaluation", "Time Series", "Advanced Analytics", "Model Comparison", "Download", "Utilities"
])

uploaded_file = st.sidebar.file_uploader("Upload CSV File", type="csv")
data = None

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.session_state["raw_data"] = data.copy()
    st.sidebar.subheader("Preprocessing")

    if st.sidebar.checkbox("Label Encode Categoricals"):
        label_encoders = {}
        for col in data.select_dtypes(include='object').columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            label_encoders[col] = le
        st.session_state["label_encoders"] = label_encoders
        st.success("Categoricals converted.")

    if st.sidebar.checkbox("Fill Missing With Mean"):
        data = data.fillna(data.mean(numeric_only=True))

    if st.sidebar.checkbox("Standard Scaling"):
        scaler = StandardScaler()
        data[data.select_dtypes(include=np.number).columns] = scaler.fit_transform(data.select_dtypes(include=np.number))

    st.session_state["clean_data"] = data
    st.dataframe(data.head())

    if menu == "Home":
        st.image("https://via.placeholder.com/800x200.png?text=Data+Science+Workbench", use_container_width=True)
        st.markdown("""
        ### Welcome to the Data Science Workbench 🚀
        - 🧠 Automated ML
        - 🧮 Explainable AI
        - ⏳ Time Series Ready
        - 📊 Real-Time Charting
        - 🔍 Outlier Detection
        - ✅ Predict with Form UI
        """)

    elif menu == "EDA":
        st.subheader("EDA Dashboard")
        st.dataframe(data.describe())
        col = st.selectbox("Choose Column to Plot", data.columns)
        st.plotly_chart(px.histogram(data, x=col))
        st.write("### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    elif menu == "Modeling":
        st.subheader("Train Model")
        target = st.selectbox("Target Variable", data.columns)
        X = data.drop(columns=[target])
        y = data[target]
        task = "Regression" if y.nunique() > 10 and np.issubdtype(y.dtype, np.number) else "Classification"

        if st.checkbox("Add Polynomial Features"):
            degree = st.slider("Degree", 2, 5, 2)
            poly = PolynomialFeatures(degree)
            X_poly = poly.fit_transform(X)
            feature_names = poly.get_feature_names_out(X.columns)
            X = pd.DataFrame(X_poly, columns=feature_names)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            "Random Forest": RandomForestClassifier() if task == "Classification" else RandomForestRegressor(),
            "Gradient Boosting": GradientBoostingClassifier() if task == "Classification" else GradientBoostingRegressor(),
            "Logistic/Linear": LogisticRegression(max_iter=1000) if task == "Classification" else LinearRegression(),
            "SVM": SVC() if task == "Classification" else SVR()
        }

        model_name = st.selectbox("Model", list(models.keys()))
        if st.button("Train"):
            model = models[model_name]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.session_state.update({"model": model, "X_test": X_test, "y_test": y_test, "features": X_train, "target_name": target})
            st.success("Model Trained")

            if task == "Classification":
                st.text(classification_report(y_test, y_pred))
                cm = confusion_matrix(y_test, y_pred)
                st.pyplot(sns.heatmap(cm, annot=True, fmt='d').figure)
            else:
                st.metric("MAE", mean_absolute_error(y_test, y_pred))
                st.metric("R2", r2_score(y_test, y_pred))

            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                importance_df = pd.DataFrame({"Feature": X.columns, "Importance": importances})
                importance_df = importance_df.sort_values(by="Importance", ascending=False)
                st.subheader("Feature Importance")
                st.plotly_chart(px.bar(importance_df, x="Feature", y="Importance", title="Feature Importances"))

    elif menu == "AutoML":
        st.subheader("AutoML with LazyPredict")
        target = st.selectbox("Select Target Column", data.columns)
        X = data.drop(columns=[target])
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        task = "Regression" if y.nunique() > 10 and y.dtypes in [float, int] else "Classification"

        if st.button("Run AutoML"):
            if task == "Classification":
                model = LazyClassifier(verbose=0, ignore_warnings=True)
            else:
                model = LazyRegressor(verbose=0, ignore_warnings=True)
            with st.spinner("Running AutoML..."):
                models, predictions = model.fit(X_train, X_test, y_train, y_test)
                st.write(models)

    elif menu == "Prediction":
        st.subheader("Make Predictions")
        if "model" in st.session_state:
            model = st.session_state.model
            features_df = st.session_state.features
            target_name = st.session_state.get("target_name", "Target")
            input_values = []
            if isinstance(features_df, pd.DataFrame):
                for col in features_df.columns:
                    val = st.number_input(f"{col}", step=0.01)
                    input_values.append(val)
                input_df = pd.DataFrame([input_values], columns=features_df.columns)
                if st.button("Predict"):
                    result = model.predict(input_df)
                    readable_result = result[0]
                    if target_name in st.session_state.get("label_encoders", {}):
                        readable_result = st.session_state.label_encoders[target_name].inverse_transform([result[0]])[0]
                    st.success(f"Prediction: {readable_result}")
                    if hasattr(model, "feature_importances_"):
                        importances = model.feature_importances_
                        importance_df = pd.DataFrame({"Feature": features_df.columns, "Importance": importances})
                        importance_df = importance_df.sort_values(by="Importance", ascending=False)
                        st.subheader("Feature Importance Explanation")
                        st.plotly_chart(px.bar(importance_df, x="Feature", y="Importance", title="Feature Contribution to Prediction"))
            else:
                st.error("Feature names not found. Please retrain the model with proper features.")

    elif menu == "Unsupervised Learning":
        st.subheader("Clustering and PCA")
        if st.checkbox("K-Means"):
            k = st.slider("# Clusters", 2, 10, 3)
            km = KMeans(n_clusters=k)
            clusters = km.fit_predict(data.select_dtypes(include=np.number))
            st.write(pd.Series(clusters).value_counts())
        if st.checkbox("DBSCAN"):
            eps = st.slider("Epsilon", 0.1, 10.0, 0.5)
            min_samples = st.slider("Min Samples", 1, 10, 5)
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(data.select_dtypes(include=np.number))
            st.write(pd.Series(db.labels_).value_counts())
        if st.checkbox("PCA Visualization"):
            pc = PCA(n_components=2).fit_transform(data.select_dtypes(include=np.number))
            st.plotly_chart(px.scatter(x=pc[:,0], y=pc[:,1]))

    elif menu == "Model Evaluation":
        if "model" in st.session_state:
            model = st.session_state.model
            scores = cross_val_score(model, st.session_state.X_test, st.session_state.y_test, cv=5)
            st.write("Cross-Validation Score:", scores.mean())

    elif menu == "Time Series":
        st.subheader("Time Series Forecasting")
        date_col = st.selectbox("Date Column", data.columns)
        value_col = st.selectbox("Value Column", [col for col in data.columns if col != date_col])
        df_ts = data[[date_col, value_col]].dropna().copy()
        df_ts[date_col] = pd.to_datetime(df_ts[date_col])
        df_ts = df_ts.set_index(date_col)
        steps = st.slider("Forecast Days", 1, 30, 7)

        model = ARIMA(df_ts, order=(5,1,0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)
        st.line_chart(pd.concat([df_ts, forecast.rename("Forecast")]))

    elif menu == "Utilities":
        st.subheader("Outlier Detection and Feature Summary")
        if st.checkbox("Show Outliers using Z-score"):
            from scipy.stats import zscore
            z_scores = np.abs(zscore(data.select_dtypes(include=np.number)))
            outliers = (z_scores > 3).sum(axis=1)
            st.write("Rows with outliers:", (outliers > 0).sum())
        if st.checkbox("Feature Skewness"):
            skewness = data.skew(numeric_only=True)
            st.write(skewness.sort_values(ascending=False))

    elif menu == "Advanced Analytics":
        st.subheader("🔬 Advanced Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Quality Report")
            if st.button("Generate Quality Report"):
                quality_report = {
                    "Total Rows": len(data),
                    "Total Columns": len(data.columns),
                    "Missing Values": data.isnull().sum().sum(),
                    "Duplicate Rows": data.duplicated().sum(),
                    "Numeric Columns": len(data.select_dtypes(include=np.number).columns),
                    "Categorical Columns": len(data.select_dtypes(include='object').columns)
                }
                st.json(quality_report)
                
                # Missing values heatmap
                if data.isnull().sum().sum() > 0:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    sns.heatmap(data.isnull(), cbar=True, ax=ax)
                    st.pyplot(fig)
        
        with col2:
            st.subheader("Statistical Summary")
            if st.button("Generate Statistical Summary"):
                numeric_data = data.select_dtypes(include=np.number)
                if not numeric_data.empty:
                    summary_stats = pd.DataFrame({
                        'Mean': numeric_data.mean(),
                        'Median': numeric_data.median(),
                        'Std': numeric_data.std(),
                        'Skewness': numeric_data.skew(),
                        'Kurtosis': numeric_data.kurtosis()
                    })
                    st.dataframe(summary_stats)
                    
                    # Distribution plots
                    selected_col = st.selectbox("Select column for distribution", numeric_data.columns)
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x=data[selected_col], name="Distribution"))
                    fig.update_layout(title=f"Distribution of {selected_col}")
                    st.plotly_chart(fig)

    elif menu == "Model Comparison":
        st.subheader("🏆 Model Performance Comparison")
        
        if "model" in st.session_state:
            target = st.selectbox("Select Target for Comparison", data.columns)
            X = data.drop(columns=[target])
            y = data[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            task_type = "Regression" if y.nunique() > 10 and np.issubdtype(y.dtype, np.number) else "Classification"
            
            if st.button("Compare All Models"):
                results = []
                
                if task_type == "Classification":
                    models_to_compare = {
                        "Random Forest": RandomForestClassifier(random_state=42),
                        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                        "Decision Tree": DecisionTreeClassifier(random_state=42),
                        "SVM": SVC(random_state=42)
                    }
                    
                    for name, model in models_to_compare.items():
                        start_time = time.time()
                        model.fit(X_train, y_train)
                        train_time = time.time() - start_time
                        
                        y_pred = model.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)
                        
                        results.append({
                            "Model": name,
                            "Accuracy": accuracy,
                            "Training Time (s)": round(train_time, 3)
                        })
                else:
                    models_to_compare = {
                        "Random Forest": RandomForestRegressor(random_state=42),
                        "Linear Regression": LinearRegression(),
                        "Gradient Boosting": GradientBoostingRegressor(random_state=42)
                    }
                    
                    for name, model in models_to_compare.items():
                        start_time = time.time()
                        model.fit(X_train, y_train)
                        train_time = time.time() - start_time
                        
                        y_pred = model.predict(X_test)
                        r2 = r2_score(y_test, y_pred)
                        mae = mean_absolute_error(y_test, y_pred)
                        
                        results.append({
                            "Model": name,
                            "R² Score": r2,
                            "MAE": mae,
                            "Training Time (s)": round(train_time, 3)
                        })
                
                results_df = pd.DataFrame(results)
                st.dataframe(results_df)
                
                # Performance visualization
                if task_type == "Classification":
                    fig = px.bar(results_df, x="Model", y="Accuracy", title="Model Accuracy Comparison")
                else:
                    fig = px.bar(results_df, x="Model", y="R² Score", title="Model R² Score Comparison")
                st.plotly_chart(fig)

    elif menu == "Download":
        st.subheader("📥 Download Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if "clean_data" in st.session_state:
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    st.session_state.clean_data.to_excel(writer, index=False)
                st.download_button("📊 Download Processed Excel", output.getvalue(), "processed_data.xlsx")
        
        with col2:
            if "clean_data" in st.session_state:
                csv_data = st.session_state.clean_data.to_csv(index=False)
                st.download_button("📄 Download CSV", csv_data, "processed_data.csv")
        
        with col3:
            if "model" in st.session_state:
                model_bytes = io.BytesIO()
                joblib.dump(st.session_state.model, model_bytes)
                st.download_button("🤖 Download Trained Model", model_bytes.getvalue(), "trained_model.pkl")

else:
    st.info("👈 Upload a dataset from the sidebar to get started.")
    
    # Sample data generation
    st.subheader("🎲 Or Generate Sample Data")
    sample_type = st.selectbox("Choose sample dataset", 
                              ["Iris", "Boston Housing", "Wine Classification", "Random Data"])
    
    if st.button("Generate Sample Data"):
        if sample_type == "Iris":
            from sklearn.datasets import load_iris
            iris = load_iris()
            data = pd.DataFrame(iris.data, columns=iris.feature_names)
            data['target'] = iris.target
            st.session_state["clean_data"] = data
            st.success("Iris dataset loaded!")
            st.dataframe(data.head())
            
        elif sample_type == "Random Data":
            np.random.seed(42)
            random_data = pd.DataFrame({
                'feature_1': np.random.randn(100),
                'feature_2': np.random.randn(100),
                'feature_3': np.random.randint(0, 5, 100),
                'target': np.random.randint(0, 2, 100)
            })
            st.session_state["clean_data"] = random_data
            st.success("Random dataset generated!")
            st.dataframe(random_data.head())