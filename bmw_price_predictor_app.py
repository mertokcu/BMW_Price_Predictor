
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle
import os

# Set page configuration
st.set_page_config(
    page_title="BMW Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3em;
        font-weight: bold;
        color: #0066CC;
        text-align: center;
        margin-bottom: 20px;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #0066CC;
        margin: 20px 0;
    }
    .success-text {
        color: #28a745;
        font-weight: bold;
        font-size: 1.2em;
    }
    .info-text {
        color: #0066CC;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_and_train_model(csv_path):
    """Load data and train the Random Forest model"""
    try:
        # Load dataset
        df = pd.read_csv(csv_path)
        
        # Feature Engineering
        current_year = df['year'].max()
        df['car_age'] = current_year - df['year']
        df['mileage_per_year'] = df['mileage'] / (df['car_age'] + 1)
        
        # Encoding categorical variables
        df_encoded = pd.get_dummies(df, columns=['model', 'transmission', 'fuelType'], drop_first=True)
        
        # Split features and target
        X = df_encoded.drop('price', axis=1)
        y = df_encoded['price']
        
        # Time-ordered split
        cutoff = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:cutoff], X.iloc[cutoff:]
        y_train, y_test = y.iloc[:cutoff], y.iloc[cutoff:]
        
        # Train Random Forest model
        rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # Train Linear Regression model
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        
        return rf_model, lr_model, X_train, df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None

def predict_car_price(model, X_train, df, model_name, year, transmission, fuel_type, mileage, tax, mpg, engine_size):
    """Make prediction for a new car"""
    try:
        # Create new car record
        new_car = pd.DataFrame({
            'model': [model_name],
            'year': [year],
            'transmission': [transmission],
            'fuelType': [fuel_type],
            'mileage': [mileage],
            'tax': [tax],
            'mpg': [mpg],
            'engineSize': [engine_size]
        })
        
        # Engineer features
        current_year = df['year'].max()
        new_car['car_age'] = current_year - new_car['year']
        new_car['mileage_per_year'] = new_car['mileage'] / (new_car['car_age'] + 1)
        
        # Encode categorical variables
        new_car_encoded = pd.get_dummies(new_car, columns=['model', 'transmission', 'fuelType'], drop_first=True)
        
        # Align with training features
        for col in X_train.columns:
            if col not in new_car_encoded.columns:
                new_car_encoded[col] = 0
        
        new_car_encoded = new_car_encoded[X_train.columns]
        
        # Make prediction
        predicted_price = model.predict(new_car_encoded)[0]
        
        return predicted_price
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

# Main app
st.markdown("<div class='main-header'>🚗 BMW Price Predictor</div>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar for model selection and settings
st.sidebar.header("⚙️ Settings")
selected_model = st.sidebar.radio("Choose Prediction Model:", 
                                 ["Random Forest (Recommended)", "Linear Regression"],
                                 help="Random Forest typically provides better predictions")

# CSV file path
csv_path = '/Users/mertokcu/Desktop/BMW/bmw.csv'

# Check if file exists
if not os.path.exists(csv_path):
    st.error(f"❌ CSV file not found at {csv_path}")
    st.stop()

# Load data and train model
with st.spinner("🔄 Loading data and training model..."):
    rf_model, lr_model, X_train, df = load_and_train_model(csv_path)
    
    if rf_model is None:
        st.stop()

# Select the appropriate model
model_to_use = rf_model if "Random Forest" in selected_model else lr_model
model_name_display = "Random Forest" if "Random Forest" in selected_model else "Linear Regression"

# Main content area
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.header("📝 Enter Car Details")
    
    # Get unique values from dataset for dropdowns
    models = sorted(df['model'].unique().tolist())
    transmissions = sorted(df['transmission'].unique().tolist())
    fuel_types = sorted(df['fuelType'].unique().tolist())
    
    # Input fields
    car_model = st.selectbox(
        "BMW Model",
        models,
        help="Select the BMW model"
    )
    
    col_a, col_b = st.columns(2)
    with col_a:
        year = st.slider(
            "Year of Manufacture",
            min_value=int(df['year'].min()),
            max_value=int(df['year'].max()),
            value=int(df['year'].max()) - 2,
            step=1,
            help="Year of the car"
        )
    
    with col_b:
        transmission = st.selectbox(
            "Transmission Type",
            transmissions,
            help="Manual or Automatic"
        )
    
    fuel_type = st.selectbox(
        "Fuel Type",
        fuel_types,
        help="Petrol, Diesel, Hybrid, or Electric"
    )
    
    col_c, col_d = st.columns(2)
    with col_c:
        mileage = st.number_input(
            "Mileage (miles)",
            min_value=0,
            max_value=500000,
            value=50000,
            step=1000,
            help="Total miles driven"
            
        )
    
    with col_d:
        tax = st.number_input(
            "Annual Tax (£)",
            min_value=0,
            max_value=500,
            value=150,
            step=10,
            help="Annual tax amount in pounds"
        )
    
    col_e, col_f = st.columns(2)
    with col_e:
        mpg = st.number_input(
            "Fuel Efficiency (MPG)",
            min_value=10.0,
            max_value=70.0,
            value=45.0,
            step=0.5,
            help="Miles per gallon"
        )
    
    with col_f:
        engine_size = st.number_input(
            "Engine Size (Liters)",
            min_value=1.0,
            max_value=6.0,
            value=2.0,
            step=0.1,
            help="Engine displacement in liters"
        )

with col2:
    st.header("🎯 Prediction Results")
    
    # Prediction button
    if st.button("🔍 Predict Price", use_container_width=True, key="predict_btn"):
        with st.spinner("Calculating prediction..."):
            predicted_price = predict_car_price(
                model_to_use,
                X_train,
                df,
                car_model,
                year,
                transmission,
                fuel_type,
                mileage,
                tax,
                mpg,
                engine_size
            )
        
        if predicted_price is not None:
            # Display prediction
            st.markdown(f"""
                <div class='prediction-box'>
                    <p style='margin: 0; font-size: 0.9em; color: #666;'>Estimated Market Price</p>
                    <p class='success-text' style='margin: 10px 0 0 0; font-size: 2.5em;'>£{predicted_price:,.2f}</p>
                    <p style='margin: 10px 0 0 0; font-size: 0.85em; color: #666;'>Using {model_name_display}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Additional information
            st.subheader("📊 Car Summary")
            summary_data = {
                "Model": car_model,
                "Year": year,
                "Age (years)": pd.Timestamp.now().year - year,
                "Transmission": transmission,
                "Fuel Type": fuel_type,
                "Mileage": f"{mileage:,} miles",
                "Annual Tax": f"£{tax}",
                "Fuel Efficiency": f"{mpg} MPG",
                "Engine Size": f"{engine_size}L",
                "Predicted Price": f"£{predicted_price:,.2f}"
            }
            
            summary_df = pd.DataFrame(list(summary_data.items()), columns=["Property", "Value"])
            st.table(summary_df)
            
            # Price context
            st.subheader("💡 Price Context")
            col_context1, col_context2, col_context3 = st.columns(3)
            
            with col_context1:
                avg_price = df['price'].mean()
                st.metric("Dataset Average Price", f"£{avg_price:,.0f}")
            
            with col_context2:
                model_avg = df[df['model'] == car_model]['price'].mean()
                st.metric(f"Avg {car_model} Price", f"£{model_avg:,.0f}")
            
            with col_context3:
                year_avg = df[df['year'] == year]['price'].mean()
                st.metric(f"Avg {year} Model Price", f"£{year_avg:,.0f}")

# Footer section
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.85em;'>
        <p><strong>Data Source:</strong> BMW Market Dataset</p>
        <p><strong>Model Type:</strong> Machine Learning (Random Forest / Linear Regression)</p>
        <p><em>Predictions are estimates based on historical market data and may vary based on actual condition and market conditions.</em></p>
    </div>
""", unsafe_allow_html=True)

# Display model info in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("📈 Model Information")
st.sidebar.info(
    f"""
    **Current Model:** {model_name_display}
    
    **Dataset Stats:**
    - Total Cars: {len(df):,}
    - Year Range: {int(df['year'].min())} - {int(df['year'].max())}
    - Price Range: £{df['price'].min():,.0f} - £{df['price'].max():,.0f}
    - Avg Price: £{df['price'].mean():,.0f}
    """
)
