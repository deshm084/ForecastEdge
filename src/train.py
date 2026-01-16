import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error
from joblib import Parallel, delayed
import warnings
import os

# --- 1. MLOps & Config ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings('ignore') # Silence Prophet warnings

FORECAST_HORIZON = 30
TRAIN_END_DATE = '2017-07-15' 

# --- 2. Data Loader Class ---
class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.raw_data = None

    def load_and_clean(self):
        """Ingests raw data and applies critical cleaning steps."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Could not find file: {self.file_path}. Make sure you are running from the ForecastEdge folder!")

        logging.info("Loading data...")
        self.raw_data = pd.read_csv(self.file_path)
        
        # Standardize Dates
        self.raw_data['date'] = pd.to_datetime(self.raw_data['date'])
        
        # Aggregate to Category Level (Family)
        daily_sales = self.raw_data.groupby(['date', 'family'])['sales'].sum().reset_index()
        
        # Filter Data Range (2015 onwards)
        daily_sales = daily_sales[daily_sales['date'] >= '2015-08-15']
        
        # This was the line causing your error - it is clean now
        logging.info(f"Data Loaded. Rows: {len(daily_sales)} | Categories: {daily_sales['family'].nunique()}")
        return daily_sales

    @staticmethod
    def get_holidays():
        """Creates a custom holiday DataFrame (Jan 1st Anomaly)."""
        holiday_dates = pd.DataFrame({
            'holiday': 'New Year Closing',
            'ds': pd.to_datetime(['2016-01-01', '2017-01-01']),
            'lower_window': 0,
            'upper_window': 1,
        })
        return holiday_dates

# --- 3. Forecast Engine Class ---
class ForecastEngine:
    def __init__(self, train_df, validation_df):
        self.train_df = train_df
        self.validation_df = validation_df
        self.models = {}
        self.results = {}
        self.holidays = DataLoader.get_holidays()

    def _train_single_category(self, category):
        """Worker function to train a model for ONE category."""
        # Prepare data for Prophet (DS/Y format)
        df_cat = self.train_df[self.train_df['family'] == category][['date', 'sales']]
        df_cat.columns = ['ds', 'y']

        # Setup Prophet
        m = Prophet(
            seasonality_mode='multiplicative',
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            holidays=self.holidays,
            changepoint_prior_scale=0.05
        )
        m.add_country_holidays(country_name='EC')
        m.fit(df_cat)
        
        # Predict
        future = m.make_future_dataframe(periods=FORECAST_HORIZON)
        forecast = m.predict(future)
        
        # Extract Result
        forecast_trim = forecast[['ds', 'yhat']].tail(FORECAST_HORIZON)
        return category, m, forecast_trim

    def train_all_categories(self):
        """Trains models for ALL categories in PARALLEL."""
        categories = self.train_df['family'].unique()
        logging.info(f"Starting parallel training for {len(categories)} categories...")
        
        results_list = Parallel(n_jobs=-1)(
            delayed(self._train_single_category)(cat) for cat in categories
        )
        
        for cat, model, forecast in results_list:
            self.models[cat] = model
            self.results[cat] = forecast
            
        logging.info("Training complete.")

    def evaluate(self):
        """Calculates WMAPE (Weighted Error)."""
        total_error = 0
        total_sales = 0
        category_metrics = []

        for cat in self.results.keys():
            predicted = self.results[cat]['yhat'].values
            actual = self.validation_df[self.validation_df['family'] == cat]['sales'].values
            
            # Metric Calculation
            # Avoid division by zero by replacing 0 with 1 for MAPE calc
            actual_safe = np.where(actual == 0, 1, actual)
            
            mape = mean_absolute_percentage_error(actual_safe, predicted)
            w_error = np.sum(np.abs(actual - predicted))
            w_sales = np.sum(actual)
            
            total_error += w_error
            total_sales += w_sales
            
            category_metrics.append({'Category': cat, 'MAPE': mape})

        if total_sales == 0:
            global_wmape = 0
        else:
            global_wmape = total_error / total_sales
            
        logging.info(f"Global WMAPE: {global_wmape:.4%}")
        return pd.DataFrame(category_metrics).sort_values(by='MAPE')

# --- 4. Main Execution ---
if __name__ == "__main__":
    # Ensure train.csv is in the 'data' folder or current folder
    # Adjust this path if your csv is inside a 'data' subfolder
    csv_path = 'train.csv' 
    
    try:
        data_loader = DataLoader(csv_path) 
        df = data_loader.load_and_clean()

        # Split Data
        train_data = df[df['date'] <= TRAIN_END_DATE]
        valid_data = df[df['date'] > TRAIN_END_DATE]
        valid_data = valid_data[valid_data['date'] <= pd.to_datetime(TRAIN_END_DATE) + pd.Timedelta(days=FORECAST_HORIZON)]

        # Run Engine
        engine = ForecastEngine(train_data, valid_data)
        engine.train_all_categories()

        # Evaluate
        metrics_df = engine.evaluate()
        print("\nTop 5 Best Performing Categories (Lowest MAPE):")
        print(metrics_df.head(5))

    except Exception as e:
        logging.error(f"An error occurred: {e}")