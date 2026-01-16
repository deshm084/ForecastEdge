import pandas as pd
import logging
import os

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.raw_data = None

    def load_and_clean(self):
        """Ingests raw data and applies critical cleaning steps."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Could not find file: {self.file_path}")

        logging.info(f"Loading data from {self.file_path}...")
        self.raw_data = pd.read_csv(self.file_path)
        
        # Standardize Dates
        self.raw_data['date'] = pd.to_datetime(self.raw_data['date'])
        
        # Aggregate to Category Level (Family)
        # Sum sales across all stores to get National Level demand
        daily_sales = self.raw_data.groupby(['date', 'family'])['sales'].sum().reset_index()
        
        # Filter Data Range (2015 onwards as per analysis)
        daily_sales = daily_sales[daily_sales['date'] >= '2015-08-15']
        
        logging.info(f"Data Loaded. Rows: {len(daily_sales)} | Categories: {daily_sales['family'].nunique()}")
        return daily_sales

    @staticmethod
    def get_holidays():
        """
        Creates a custom holiday DataFrame.
        CRITICAL: We manually flag Jan 1st as a 'holiday' so Prophet ignores the zero sales.
        """
        holiday_dates = pd.DataFrame({
            'holiday': 'New Year Closing',
            'ds': pd.to_datetime(['2016-01-01', '2017-01-01']),
            'lower_window': 0,
            'upper_window': 1,
        })
        return holiday_dates
