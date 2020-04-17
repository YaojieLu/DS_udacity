
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

# import csv
cal = pd.read_csv('../Data/Project_1/calendar.csv')

# income analysis
cal['price'] = cal.apply(lambda row: float(re.sub('(\$|,)', '', row['price'])) if isinstance(row['price'], str) else np.nan, axis=1)
cal['price'] = cal['price'].ffill().bfill()
cal['available'] = cal['available'].replace(['t', 'f'], [0, 1])
price = cal['price']
price = pd.DataFrame(price.values.reshape(365, -1))
occupancy = pd.DataFrame(cal['available'].values.reshape(365, -1))
income = price*occupancy
income_sum = income.sum()
id = list(cal['listing_id'].values.reshape(-1, 365)[:, 0])

# figure
fig = plt.figure(figsize=(10, 10))
plt.hist(cal[cal['available']==0]['price'], bins=10, alpha=0.5, density=True, label='available')
plt.hist(cal[cal['available']==1]['price'], bins=10, alpha=0.5, density=True, label='not available')
plt.legend(loc='upper right', fontsize=20)
plt.xlabel('Listing price', fontsize=20)
plt.tick_params(labelsize=15)
plt.savefig('Figures/Figure 5.png', bbox_inches='tight')
