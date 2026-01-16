import numpy as np
import pandas as pd
import logging
from sklearn.metrics import mean_absolute_percentage_error

class Evaluator:
    @staticmethod
    def calculate_metrics(results_dict, validation_df):
        """
        Calculates WMAPE (Weighted Error) across all categories.
        
        Args:
            results_dict: Dict {category_name: forecast_dataframe}
            validation_df: The actual validation dataframe
        """
        total_error = 0
        total_sales = 0
        category_metrics = []

        logging.info("Starting evaluation...")

        for cat, forecast_df in results_dict.items():
            # Align Dates: Get the predicted values for the validation period
            # We assume the last 30 days of the forecast match the validation set
            predicted = forecast_df['yhat'].values[-30:] 
            actual = validation_df[validation_df['family'] == cat]['sales'].values
            
            # Safety check: Ensure lengths match
            min_len = min(len(predicted), len(actual))
            predicted = predicted[:min_len]
            actual = actual[:min_len]

            # Avoid division by zero for MAPE
            actual_safe = np.where(actual == 0, 1, actual)
            
            # Metric Calculation
            mape = mean_absolute_percentage_error(actual_safe, predicted)
            w_error = np.sum(np.abs(actual - predicted))
            w_sales = np.sum(actual)
            
            total_error += w_error
            total_sales += w_sales
            
            category_metrics.append({'Category': cat, 'MAPE': mape})

        # Calculate Global WMAPE
        if total_sales == 0:
            global_wmape = 0
        else:
            global_wmape = total_error / total_sales
            
        logging.info(f"Global WMAPE: {global_wmape:.4%}")
        
        return global_wmape, pd.DataFrame(category_metrics).sort_values(by='MAPE')
