# Real Estate Price Predictor

A Streamlit web application that predicts house prices and price categories using machine learning models trained on a real estate dataset.

## Features

- **Data Overview**: View dataset statistics and sample data
- **Visualizations**: Explore price distributions, correlations, and location analysis
- **Price Prediction**: Input property details and get instant price predictions
- **Price Category Classification**: Classify properties as High, Medium, or Low value
- **Model Performance Metrics**: R² score, RMSE, and classification accuracy

## Installation (Local)

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd DS-8
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

4. Open your browser and navigate to `http://localhost:8501`

## Deployment on Streamlit Cloud

### Step 1: Push to GitHub

1. Create a new GitHub repository
2. Initialize git in your project:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Real Estate Price Predictor"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git push -u origin main
   ```

### Step 2: Deploy on Streamlit Cloud

1. Go to [Streamlit Cloud](https://streamlit.io/cloud)
2. Click "New app"
3. Select your GitHub repository and branch
4. Choose the main file: `app.py`
5. Click "Deploy"

The app will be live at `https://<your-username>-<repo-name>.streamlit.app`

## Project Files

- `app.py`: Main Streamlit application
- `regression_models.ipynb`: Jupyter notebook with regression models and analysis
- `real_estate_dataset.csv`: Real estate dataset with property features and prices
- `requirements.txt`: Python dependencies

## How It Works

### Models Used

1. **Linear Regression**: Predicts house price based on property features
2. **Decision Tree Classifier**: Categorizes properties into High, Medium, or Low price categories

### Input Features

- Area (Square Feet)
- Number of Bedrooms
- Number of Bathrooms
- Location Quality (Excellent, Good, Average, Low)
- Property Age (Years)
- Parking Availability (Yes/No)

### Output

- **Predicted Price**: Estimated property price in ₹ (Indian Rupees)
- **Price Category**: Classification (High/Medium/Low)
- **Model Accuracy**: R² score and RMSE metrics

## Dataset

The application uses the **Real Estate Dataset** containing:
- 1000+ property records
- 12 features including area, bedrooms, bathrooms, location score, and price
- Price range: ₹300,000 to ₹900,000

## Model Performance

- Linear Regression R²: ~0.85
- Classification Accuracy: ~92%

## Author

Data Scientist & Streamlit Developer

## License

MIT
