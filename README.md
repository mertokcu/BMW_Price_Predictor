# BMW_Price_Predictor
An end-to-end machine learning pipeline and real-time Streamlit web application for predicting the depreciation and resale value of used BMW vehicles.

## 📌 Project Overview
The luxury automotive market experiences rapid but highly variable depreciation. Accurately estimating the fair market value of a used BMW is a complex challenge influenced by non-linear factors like mileage, fuel type, engine size, and specific model variants. 

This project tackles this regression problem by predicting the resale price of used BMWs. It showcases an end-to-end data science lifecycle: transitioning from exploratory data analysis and basic script-based modeling to a **production-grade Scikit-Learn Pipeline**, and finally deploying it as a **real-time Streamlit web application**.

## 📊 Dataset
The project utilizes historical sales data for BMW vehicles (`bmw.csv`). 
* **Target Variable:** `price` (in GBP)
* **Features:** `model`, `year`, `transmission`, `mileage`, `fuelType`, `tax`, `mpg`, `engineSize`.

## ⚙️ Methodology & Features

### 1. Feature Engineering
To better capture vehicle wear and depreciation, two custom features were engineered:
* **`car_age`**: Calculated relative to the current context year.
* **`annual_mileage`**: A proxy for usage intensity (highway cruiser vs. city car).

### 2. Time-Series Split
To prevent data leakage, the dataset was explicitly sorted by `year`. The oldest 80% of cars were used for training, while the newest 20% were held out for testing. This simulates the real-world challenge of using historical data to predict the value of newer cars entering the market.

### 3. Production-Grade Pipeline
The final model utilizes a `scikit-learn` Pipeline to ensure robust data handling:
* **Imputation:** Handles missing data dynamically (`median` for numerical, `most_frequent` for categorical).
* **Robust Encoding:** Replaced `pd.get_dummies` with `OneHotEncoder(handle_unknown='ignore')` to prevent crashes if unseen car models appear in production.
* **Algorithm:** `RandomForestRegressor` (300 estimators) was chosen for its ability to handle non-linear depreciation curves and complex feature interactions.

## 📈 Results
The Random Forest model significantly outperformed the baseline Linear Regression model, successfully capturing the non-linear "cliffs" in luxury car depreciation.
* **Metrics Tracked:** Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-Squared ($R^2$).
* The pipeline accurately prices vehicles within a narrow error margin, explaining the vast majority of variance in the test set.

## 🚀 Web Application (Streamlit)
The project includes a real-time prediction app (`app.py`). Users can input vehicle specifications via a clean UI, and the app dynamically aligns the inputs, engineers the necessary features, and queries the serialized model pipeline to estimate the car's resale value.
