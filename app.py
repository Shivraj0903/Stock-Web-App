import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Try to import Keras with error handling
try:
    from keras.models import load_model
    keras_available = True
except ImportError as e:
    st.error(f"Keras/TensorFlow not available: {e}")
    keras_available = False

# Configure the page
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

def main():
    st.title("📈 Stock Market Prediction")
    
    # Sidebar for user inputs
    st.sidebar.header("Stock Selection")
    user_input = st.sidebar.text_input('Enter Stock Ticker', 'AAPL').upper()
    
    # Date range selection
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input('Start Date', pd.to_datetime('2010-01-01'))
    with col2:
        end_date = st.date_input('End Date', pd.to_datetime('2024-03-31'))
    
    if user_input:
        try:
            # Download stock data
            with st.spinner('Downloading stock data...'):
                df = yf.download(user_input, start=start_date, end=end_date)
            
            if df.empty:
                st.error('No data found for the given ticker. Please try another.')
                return
            
            display_data_analysis(df, user_input)
            
            if keras_available:
                make_predictions(df, user_input)
            else:
                st.warning("Model prediction disabled - Keras/TensorFlow not available")
                
        except Exception as e:
            st.error(f'An error occurred: {str(e)}')
    else:
        st.warning('Please enter a stock ticker.')

def display_data_analysis(df, user_input):
    """Display data analysis and visualizations"""
    
    # Data overview
    st.subheader(f'Data Overview for {user_input}')
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Days", len(df))
    with col2:
        st.metric("Current Price", f"${df['Close'][-1]:.2f}")
    with col3:
        st.metric("50-Day Average", f"${df['Close'].rolling(50).mean()[-1]:.2f}")
    with col4:
        price_change = ((df['Close'][-1] - df['Close'][0]) / df['Close'][0]) * 100
        st.metric("Total Return", f"{price_change:.2f}%")
    
    # Visualization section
    st.subheader('Price Charts')
    
    # Create tabs for different charts
    tab1, tab2, tab3 = st.tabs(["Closing Price", "Moving Averages", "Price Range"])
    
    with tab1:
        fig = plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['Close'], linewidth=2)
        plt.title(f'{user_input} Closing Price Over Time', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Closing Price (USD)', fontsize=12)
        plt.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with tab2:
        fig = plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['Close'], label='Closing Price', alpha=0.7)
        plt.plot(df.index, df['Close'].rolling(50).mean(), label='50-Day MA', color='orange')
        plt.plot(df.index, df['Close'].rolling(200).mean(), label='200-Day MA', color='red')
        plt.title(f'{user_input} with Moving Averages', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price (USD)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        st.pyplot(fig)

def make_predictions(df, user_input, train_split=0.7):
    """Make stock price predictions using the trained model"""
    
    st.subheader('📊 Stock Price Prediction')
    
    try:
        # Load the model
        with st.spinner('Loading prediction model...'):
            model = load_model('keras_model.h5')
        
        # Prepare data
        data = df.filter(['Close'])
        dataset = data.values
        
        # Split data
        training_data_len = int(len(dataset) * train_split)
        
        data_training = data[:training_data_len]
        data_testing = data[training_data_len:]
        
        # Scale data
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_training_array = scaler.fit_transform(data_training)
        
        # Prepare testing data
        past_100_days = data_training.tail(100)
        final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
        input_data = scaler.transform(final_df)
        
        # Create test datasets
        x_test = []
        y_test = []
        
        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i-100: i])
            y_test.append(input_data[i, 0])
        
        x_test, y_test = np.array(x_test), np.array(y_test)
        
        # Make predictions
        with st.spinner('Making predictions...'):
            y_predicted = model.predict(x_test)
        
        # Inverse transform predictions
        scale_factor = 1 / scaler.scale_[0]
        y_predicted = y_predicted * scale_factor
        y_test = y_test * scale_factor
        
        # Display predictions
        col1, col2 = st.columns(2)
        with col1:
            latest_actual = y_test[-1]
            latest_predicted = y_predicted[-1][0]
            
            st.metric("Actual Price", f"${latest_actual:.2f}")
            st.metric("Predicted Price", f"${latest_predicted:.2f}")
        
        # Plot predictions
        fig = plt.figure(figsize=(12, 6))
        plt.plot(y_test, 'b-', label='Actual Price', linewidth=2)
        plt.plot(y_predicted, 'r-', label='Predicted Price', linewidth=2, alpha=0.8)
        plt.title(f'{user_input} - Actual vs Predicted Prices', fontsize=16, fontweight='bold')
        plt.xlabel('Time (Days)', fontsize=12)
        plt.ylabel('Price (USD)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.info("Make sure you have the 'keras_model.h5' file in your project directory.")

if __name__ == "__main__":
    main()
