import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

MODEL_PATH = "model.pkl"
DATA_PATH = "real_estate_dataset.csv"

sns.set(style="whitegrid", palette="muted", font_scale=1.05)


@st.cache_data
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def create_location_category(score: float) -> str:
    if score >= 7.5:
        return "Excellent"
    if score >= 5.0:
        return "Good"
    if score >= 2.5:
        return "Average"
    return "Low"


def preprocess_data(df: pd.DataFrame, training: bool = True):
    df = df.copy()
    current_year = datetime.now().year
    df["Property_Age"] = current_year - df["Year_Built"]
    df["Location_Category"] = df["Location_Score"].apply(create_location_category)
    df["Parking"] = (df["Garage_Size"] > 0).astype(int)

    features = [
        "Square_Feet",
        "Num_Bedrooms",
        "Num_Bathrooms",
        "Property_Age",
        "Location_Category",
        "Parking",
    ]
    X = df[features]

    if training:
        price_category = pd.qcut(
            df["Price"], q=3, labels=["Low", "Medium", "High"]
        )
        return X, df["Price"], price_category
    return X


def build_preprocessor() -> ColumnTransformer:
    numeric_features = [
        "Square_Feet",
        "Num_Bedrooms",
        "Num_Bathrooms",
        "Property_Age",
        "Parking",
    ]
    categorical_features = ["Location_Category"]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def train_models(df: pd.DataFrame) -> dict:
    X, y_price, y_category = preprocess_data(df, training=True)
    preprocessor = build_preprocessor()
    X_processed = preprocessor.fit_transform(X)

    regressor = LinearRegression()
    regressor.fit(X_processed, y_price)

    classifier = DecisionTreeClassifier(max_depth=5, random_state=42)
    classifier.fit(X_processed, y_category)

    y_price_pred = regressor.predict(X_processed)
    y_category_pred = classifier.predict(X_processed)

    metrics = {
        "r2": r2_score(y_price, y_price_pred),
        "rmse": np.sqrt(mean_squared_error(y_price, y_price_pred)),
        "accuracy": accuracy_score(y_category, y_category_pred),
    }

    model_data = {
        "preprocessor": preprocessor,
        "regressor": regressor,
        "classifier": classifier,
        "metrics": metrics,
    }
    return model_data


def save_model(model_data: dict, path: str = MODEL_PATH) -> None:
    with open(path, "wb") as f:
        pickle.dump(model_data, f)


def load_model(path: str = MODEL_PATH) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_or_train_model(data_path: str = DATA_PATH, model_path: str = MODEL_PATH) -> dict:
    if os.path.exists(model_path):
        try:
            return load_model(model_path)
        except Exception:
            pass

    df = load_data(data_path)
    model_data = train_models(df)
    save_model(model_data, model_path)
    return model_data


def format_currency(value: float) -> str:
    return f"₹{value:,.0f}"


def get_category_label(category: str) -> tuple[str, str]:
    if category == "High":
        return "High", "error"
    if category == "Medium":
        return "Medium", "warning"
    return "Low", "success"


def make_prediction(
    regressor,
    classifier,
    preprocessor,
    area: float,
    bedrooms: int,
    bathrooms: int,
    location_category: str,
    property_age: int,
    parking: int,
) -> tuple[float, str]:
    user_df = pd.DataFrame(
        [
            {
                "Square_Feet": area,
                "Num_Bedrooms": bedrooms,
                "Num_Bathrooms": bathrooms,
                "Property_Age": property_age,
                "Location_Category": location_category,
                "Parking": parking,
            }
        ]
    )
    X_user = preprocessor.transform(user_df)
    price_pred = regressor.predict(X_user)[0]
    category_pred = classifier.predict(X_user)[0]
    return price_pred, category_pred


def plot_visualizations(df: pd.DataFrame) -> None:
    df_viz = df.copy()
    df_viz["Location_Category"] = df_viz["Location_Score"].apply(create_location_category)
    df_viz["Property_Age"] = datetime.now().year - df_viz["Year_Built"]

    fig_price = plt.figure()
    sns.histplot(df_viz["Price"], bins=30, kde=True)
    plt.title("Distribution of Property Prices")
    plt.xlabel("Price")
    st.pyplot(fig_price)

    fig_scatter = plt.figure()
    sns.scatterplot(x="Square_Feet", y="Price", hue="Location_Category", data=df_viz, palette="tab10")
    plt.title("Area vs Price")
    st.pyplot(fig_scatter)

    avg_price_by_loc = df_viz.groupby("Location_Category")["Price"].mean().reset_index()
    fig_bar = plt.figure()
    sns.barplot(x="Location_Category", y="Price", data=avg_price_by_loc, palette="crest")
    plt.title("Average Price by Location Category")
    plt.xlabel("Location Category")
    plt.ylabel("Average Price")
    st.pyplot(fig_bar)

    corr_columns = [
        "Square_Feet",
        "Num_Bedrooms",
        "Num_Bathrooms",
        "Num_Floors",
        "Year_Built",
        "Garage_Size",
        "Location_Score",
        "Distance_to_Center",
        "Price",
    ]
    corr = df_viz[corr_columns].corr()
    fig_heatmap = plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Heatmap")
    st.pyplot(fig_heatmap)


def main():
    st.set_page_config(page_title="Real Estate Price Predictor", layout="wide")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Home", "Data Overview", "Visualizations", "Prediction"],
    )

    df = load_data(DATA_PATH)
    model_data = load_or_train_model(DATA_PATH, MODEL_PATH)
    metrics = model_data["metrics"]

    if page == "Home":
        st.title("Real Estate Price Predictor")
        st.markdown(
            "This application allows users to explore a real estate dataset and predict house prices and a price category. "
            "The model is trained on historical property data and uses area, bedrooms, bathrooms, age, location, and parking information."
        )
        st.markdown("### Purpose")
        st.write(
            "- Predict the expected price of a property based on user inputs."
            "\n- Classify properties into High, Medium, or Low price categories."
            "\n- Provide quick visual insight into the dataset and model performance."
        )
        st.markdown("### Dataset Used")
        st.write("Real Estate Dataset containing property attributes such as square footage, bedrooms, bathrooms, year built, location score, and price.")
        st.markdown("### Features Considered")
        st.write(
            "- Area (Square Feet)\n- Bedrooms\n- Bathrooms\n- Property Age\n- Location Quality\n- Parking availability"
        )
        st.markdown("---")
        st.markdown("## Model Summary")
        st.write(f"**Regression R²:** {metrics['r2']:.3f}")
        st.write(f"**Regression RMSE:** {metrics['rmse']:.0f}")
        st.write(f"**Classification accuracy:** {metrics['accuracy']:.3f}")

    elif page == "Data Overview":
        st.title("Data Overview")
        st.write("### Dataset Preview")
        st.dataframe(df.head())
        st.write("### Dataset Shape")
        st.write(df.shape)
        st.write("### Columns")
        st.write(list(df.columns))
        st.write("### Summary Statistics")
        st.dataframe(df.describe())

    elif page == "Visualizations":
        st.title("Visualizations")
        plot_visualizations(df)

    elif page == "Prediction":
        st.title("Predict Property Price")
        with st.sidebar.expander("Prediction Instructions"):
            st.write(
                "Enter property details and click Predict Price."
                " Larger area, more bedrooms, better location, and newer properties generally increase price."
            )

        col1, col2 = st.columns(2)
        with col1:
            area = st.number_input("Area (sq ft)", min_value=100.0, max_value=5000.0, value=1200.0, step=10.0)
            bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3, step=1)
            bathrooms = st.number_input("Bathrooms", min_value=1, max_value=5, value=2, step=1)
        with col2:
            location_category = st.selectbox(
                "Location Quality",
                ["Excellent", "Good", "Average", "Low"],
                index=1,
            )
            property_age = st.slider("Property Age (years)", min_value=0, max_value=100, value=15)
            parking = st.radio("Parking Available", ["Yes", "No"], index=0)

        parking_value = 1 if parking == "Yes" else 0

        if st.button("Predict Price"):
            with st.spinner("Predicting..."):
                price_pred, category_pred = make_prediction(
                    model_data["regressor"],
                    model_data["classifier"],
                    model_data["preprocessor"],
                    area,
                    bedrooms,
                    bathrooms,
                    location_category,
                    property_age,
                    parking_value,
                )

            formatted_price = format_currency(price_pred)
            label, severity = get_category_label(category_pred)

            if severity == "success":
                st.success(f"Predicted Price: {formatted_price}")
            elif severity == "warning":
                st.warning(f"Predicted Price: {formatted_price}")
            else:
                st.error(f"Predicted Price: {formatted_price}")

            st.write(f"### Price Category: {label}")
            st.write(
                "Larger area and better location increase property price."
                " Newer properties with parking also tend to command higher values."
            )

            result_df = pd.DataFrame(
                {
                    "Feature": [
                        "Area (sq ft)",
                        "Bedrooms",
                        "Bathrooms",
                        "Location Quality",
                        "Property Age",
                        "Parking",
                        "Predicted Price",
                        "Predicted Category",
                    ],
                    "Value": [
                        area,
                        bedrooms,
                        bathrooms,
                        location_category,
                        property_age,
                        parking,
                        formatted_price,
                        label,
                    ],
                }
            )
            st.table(result_df)
            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Prediction Result",
                data=csv,
                file_name="real_estate_prediction.csv",
                mime="text/csv",
            )

            st.markdown("---")
            st.write("### Model performance on training data")
            st.write(f"R² score: {metrics['r2']:.3f}")
            st.write(f"RMSE: {metrics['rmse']:.0f}")
            st.write(f"Classification accuracy: {metrics['accuracy']:.3f}")


if __name__ == "__main__":
    main()
