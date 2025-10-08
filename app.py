import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Configure the page
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0px;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">📈 Stock Market Prediction</h1>', unsafe_allow_html=True)
    
    # Sidebar for user inputs
    st.sidebar.header("Stock Selection")
    user_input = st.sidebar.text_input('Enter Stock Ticker', 'AAPL').upper()
    
    # Date range selection
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input('Start Date', pd.to_datetime('2010-01-01'))
    with col2:
        end_date = st.date_input('End Date', pd.to_datetime('2024-03-31'))
    
    # Additional options
    st.sidebar.header("Model Settings")
    train_split = st.sidebar.slider('Training Data Split', 0.6, 0.9, 0.7)
    
    if st.sidebar.button('Predict Stock Price'):
        if user_input:
            try:
                # Download stock data
                with st.spinner('Downloading stock data...'):
                    df = yf.download(user_input, start=start_date, end=end_date)
                
                if df.empty:
                    st.error('No data found for the given ticker. Please try another.')
                    return
                
                display_data_analysis(df, user_input)
                make_predictions(df, user_input, train_split)
                
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
    
    # Data description
    st.subheader('Statistical Summary')
    st.write(df.describe())
    
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
    
    with tab3:
        fig = plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['High'], label='High', alpha=0.7, color='green')
        plt.plot(df.index, df['Low'], label='Low', alpha=0.7, color='red')
        plt.fill_between(df.index, df['Low'], df['High'], alpha=0.3, color='gray')
        plt.title(f'{user_input} Daily Price Range', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price (USD)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        st.pyplot(fig)

def make_predictions(df, user_input, train_split):
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
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            latest_actual = y_test[-1]
            latest_predicted = y_predicted[-1][0]
            accuracy = (1 - abs(latest_actual - latest_predicted) / latest_actual) * 100
            
            st.metric("Actual Price", f"${latest_actual:.2f}")
            st.metric("Predicted Price", f"${latest_predicted:.2f}")
            st.metric("Prediction Accuracy", f"{accuracy:.2f}%")
        
        with col2:
            price_diff = latest_predicted - latest_actual
            st.metric("Price Difference", f"${price_diff:.2f}")
            if price_diff > 0:
                st.success("📈 Bullish Signal")
            else:
                st.warning("📉 Bearish Signal")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
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
        
        # Future predictions (next 30 days)
        st.subheader('🔮 Future Price Forecast')
        future_days = st.slider('Days to Predict', 1, 30, 7)
        
        if st.button('Generate Future Forecast'):
            with st.spinner('Generating future forecast...'):
                future_predictions = predict_future_prices(model, data, scaler, future_days)
                
                fig_future = plt.figure(figsize=(12, 6))
                plt.plot(range(len(y_test)), y_test, 'b-', label='Historical Actual', linewidth=2)
                plt.plot(range(len(y_test)), y_predicted.flatten(), 'r-', label='Historical Predicted', linewidth=2, alpha=0.7)
                
                future_range = range(len(y_test), len(y_test) + future_days)
                plt.plot(future_range, future_predictions, 'g--', label=f'Next {future_days} Days Forecast', linewidth=2, marker='o')
                
                plt.title(f'{user_input} - Price Forecast', fontsize=16, fontweight='bold')
                plt.xlabel('Time (Days)', fontsize=12)
                plt.ylabel('Price (USD)', fontsize=12)
                plt.legend()
                plt.grid(True, alpha=0.3)
                st.pyplot(fig_future)
                
                # Display future predictions in a table
                future_df = pd.DataFrame({
                    'Day': range(1, future_days + 1),
                    'Predicted Price': [f"${x:.2f}" for x in future_predictions]
                })
                st.write("Future Price Predictions:")
                st.dataframe(future_df)
    
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.info("Make sure you have the 'keras_model.h5' file in your project directory.")

def predict_future_prices(model, data, scaler, days=7):
    """Predict future stock prices"""
    last_100_days = data[-100:].values
    last_100_days_scaled = scaler.transform(last_100_days)
    
    future_predictions = []
    current_batch = last_100_days_scaled.reshape(1, 100, 1)
    
    for _ in range(days):
        current_pred = model.predict(current_batch, verbose=0)[0]
        future_predictions.append(current_pred[0])
        
        # Update the batch
        current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)
    
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions = scaler.inverse_transform(future_predictions)
    
    return future_predictions.flatten()

if __name__ == "__main__":
    main()