import logging
from prophet import Prophet
from joblib import Parallel, delayed
import pandas as pd
from src.data_loader import DataLoader

class ForecastEngine:
    def __init__(self, train_df, forecast_horizon=30):
        self.train_df = train_df
        self.forecast_horizon = forecast_horizon
        self.models = {}
        self.results = {}
        # Get holiday data from the loader
        self.holidays = DataLoader.get_holidays()

    def _train_single_category(self, category):
        """
        Worker function to train a model for ONE category.
        This runs inside a parallel process.
        """
        # 1. Prepare data for Prophet (DS/Y format)
        df_cat = self.train_df[self.train_df['family'] == category][['date', 'sales']]
        df_cat.columns = ['ds', 'y']

        # 2. Setup Prophet with Enterprise Tuning
        # multiplicative = sales grow with trend
        m = Prophet(
            seasonality_mode='multiplicative',
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            holidays=self.holidays,
            changepoint_prior_scale=0.05 
        )
        
        # Add Country Holidays (Ecuador specific)
        m.add_country_holidays(country_name='EC')
        
        # 3. Fit Model
        m.fit(df_cat)
        
        # 4. Predict Future
        future = m.make_future_dataframe(periods=self.forecast_horizon)
        forecast = m.predict(future)
        
        # Return key identifiers and result
        return category, m, forecast

    def train_all_categories(self):
        """
        Trains models for ALL categories in PARALLEL.
        """
        categories = self.train_df['family'].unique()
        logging.info(f"Starting parallel training for {len(categories)} categories...")
        
        # Parallel Execution: n_jobs=-1 uses all CPU cores
        results_list = Parallel(n_jobs=-1)(
            delayed(self._train_single_category)(cat) for cat in categories
        )
        
        # Unpack results into dictionary for easy access
        for cat, model, forecast in results_list:
            self.models[cat] = model
            self.results[cat] = forecast
            
        logging.info("Training complete.")
        return self.results
