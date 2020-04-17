
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score

# import csv
cal = pd.read_csv('../Data/Project_1/calendar.csv')
lis = pd.read_csv('../Data/Project_1/listings.csv')
cal['price'] = cal.apply(lambda row: float(re.sub('(\$|,)', '', row['price'])) if isinstance(row['price'], str) else np.nan, axis=1)
id = list(cal['listing_id'].values.reshape(-1, 365)[:, 0])

# analysis
price = cal['price']
price = pd.DataFrame(price.values.reshape(365, -1))
price = price.ffill().bfill()
price_mean = price.mean()
occupancy = pd.DataFrame(cal['available'].values.reshape(365, -1))
occupancy = occupancy.replace(['t', 'f'], [0, 1])
occupancy_sum = occupancy.sum()
income = price*occupancy
income_sum = income.sum()
date = cal['date'][:365]
lis = lis[['id', 'neighbourhood_cleansed']]
lis.columns = ['listing_id', 'neighborhood']

# merge
lis.set_index('listing_id', inplace=True)
df = pd.concat([price_mean, occupancy_sum, income_sum], axis=1, sort=False)
df['listing_id'] = id
df = pd.merge(df, lis, on='listing_id')
df.columns = ['price', 'occupancy', 'income', 'listing_id', 'neighborhood']

# analysis
df2 = df.groupby('neighborhood')[['price', 'occupancy', 'income']].mean()

# linear regression
v1, v2, v3 = df2[['price']], df2[['occupancy']], df2[['income']]
regr = linear_model.LinearRegression()
regr.fit(v1, v3)
v3_pred_v1 = regr.predict(v1)
r2_v3_pred_v1 = r2_score(v3, v3_pred_v1)
regr = linear_model.LinearRegression()
regr.fit(v2, v3)
v3_pred_v2 = regr.predict(v2)
r2_v3_pred_v2 = r2_score(v3, v3_pred_v2)
regr = linear_model.LinearRegression()
regr.fit(v2, v1)
v1_pred_v2 = regr.predict(v2)
r2_v1_pred_v2 = r2_score(v1, v1_pred_v2)
regr = linear_model.LinearRegression()
regr.fit(v1.join(v2), v3)
v3_pred_v1_v2 = regr.predict(v1.join(v2))
r2_v3_pred_v1_v2 = r2_score(v3, v3_pred_v1_v2)
print('R2 of the complete model = {:0.2f}'.format(r2_v3_pred_v1_v2))

# figure
fig = plt.figure(figsize=(15, 5))
# income vs. price
ax1 = fig.add_subplot(1, 3, 1)
ax1.scatter(v1, v3)
ax1.plot(v1, v3_pred_v1, 'k')
ax1.annotate('$R^2$ = {:0.2f}'.format(r2_v3_pred_v1), xy=(0.05, 0.9), xycoords='axes fraction', fontsize=15)
ax1.set_xlabel('Mean price', fontsize=20)
ax1.set_ylabel('Mean income', fontsize=20)
ax1.tick_params(labelsize=15)
# occupancy vs. price
ax2 = fig.add_subplot(1, 3, 2)
ax2.scatter(v2, v3)
ax2.plot(v2, v3_pred_v2, 'k')
ax2.annotate('$R^2$ = {:0.2f}'.format(r2_v3_pred_v2), xy=(0.05, 0.9), xycoords='axes fraction', fontsize=15)
ax2.set_xlabel('Mean occupancy', fontsize=20)
ax2.set_ylabel('Mean income', fontsize=20)
ax2.tick_params(labelsize=15)
# occupancy vs. price
ax3 = fig.add_subplot(1, 3, 3)
ax3.scatter(v2, v1)
ax3.plot(v2, v1_pred_v2, 'k')
ax3.annotate('$R^2$ = {:0.2f}'.format(r2_v1_pred_v2), xy=(0.6, 0.9), xycoords='axes fraction', fontsize=15)
ax3.set_xlabel('Mean occupancy', fontsize=20)
ax3.set_ylabel('Mean price', fontsize=20)
ax3.tick_params(labelsize=15)
plt.subplots_adjust(wspace=0.4)
plt.savefig('Figures/Figure 3.png', bbox_inches='tight')
