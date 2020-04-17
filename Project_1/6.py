
import numpy as np
import pandas as pd
import re
from textblob import TextBlob 
import matplotlib.pyplot as plt

# import csv
cal = pd.read_csv('../Data/Project_1/calendar.csv')
lis = pd.read_csv('../Data/Project_1/listings.csv')

# income analysis
cal['price'] = cal.apply(lambda row: float(re.sub('(\$|,)', '', row['price'])) if isinstance(row['price'], str) else np.nan, axis=1)
price = cal['price']
price = pd.DataFrame(price.values.reshape(365, -1))
price = price.ffill().bfill()
occupancy = pd.DataFrame(cal['available'].values.reshape(365, -1))
occupancy = occupancy.replace(['t', 'f'], [0, 1])
income = price*occupancy
income_sum = income.sum()
id = list(cal['listing_id'].values.reshape(-1, 365)[:, 0])
df = pd.DataFrame({'listing_id': id, 'total_income': income_sum,
                   'total_occupancy': occupancy.sum(), 'mean_price': price.mean()})
# review
lis = lis[['id', 'review_scores_rating']]
lis.columns = ['listing_id', 'review']

# merge
df = pd.merge(df, lis, on='listing_id')

# figure
fig = plt.figure(figsize=(10, 10))
# reivew vs. income
ax1 = fig.add_subplot(2, 1, 1)
df.plot.scatter('review', 'total_income', ax=ax1)
ax1.set_ylabel('Total income', fontsize=20)
ax1.tick_params(labelsize=15)
ax1.axes.get_xaxis().set_visible(False)
# reivew vs. occupancy
ax2 = fig.add_subplot(2, 1, 2)
df.plot.scatter('review', 'total_occupancy', ax=ax2)
ax2.set_xlabel('Review score', fontsize=20)
ax2.set_ylabel('Total occupancy', fontsize=20)
ax2.tick_params(labelsize=15)
plt.subplots_adjust(hspace=0.05)
plt.savefig('Figures/Figure 6.png', bbox_inches='tight')
