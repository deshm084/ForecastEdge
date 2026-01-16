import pandas as pd 
import numpy as np 
 
dates = pd.date_range(start='2015-01-01', end='2017-08-31') 
families = ['PRODUCE', 'AUTOMOTIVE', 'GROCERY_I', 'BEVERAGES', 'BREAD/BAKERY'] 
store_nbrs = [1, 2] 
data = [] 
print("Generating mock enterprise data...") 
for date in dates: 
    for family in families: 
        for store in store_nbrs: 
            time_idx = date.dayofyear 
            seasonality = 100 + 50 * np.sin(2 * np.pi * time_idx / 365) 
            if date.weekday() >= 5: 
                seasonality *= 1.2 
            sales = max(0, int(seasonality + np.random.normal(0, 10))) 
            data.append({'id': 0, 'date': date, 'store_nbr': store, 'family': family, 'sales': sales, 'onpromotion': 0}) 
df = pd.DataFrame(data) 
df.to_csv('train.csv', index=False) 
print("Success! Generated 'train.csv' with " + str(len(df)) + " rows.")
