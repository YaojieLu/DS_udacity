
import numpy as np
import pandas as pd
import re
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker

# import csv
cal = pd.read_csv('../Data/Project_1/calendar.csv')
#lis = pd.read_csv('../Data/Project_1/listings.csv')
#rev = pd.read_csv('../Data/Project_1/reviews.csv')

# analysis
cal['price'] = cal.apply(lambda row: float(re.sub('(\$|,)', '', row['price'])) if isinstance(row['price'], str) else np.nan, axis=1)
cal1 = cal.groupby(['date'])['price'].agg(['mean', 'std']).reset_index()
cal1 = pd.concat([cal1, pd.DataFrame(cal1['date'].str.split('-').tolist(),
                                     columns=['year', 'month', 'day'])], axis=1, sort=False)
price = pd.DataFrame(cal1['mean'][:-1].values.reshape(-1, 7))
price_week_mean = price.mean(axis=1)
price_week_std = price.std(axis=1)
weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
mondays = cal1['date'][:-1].values.reshape(-1, 7)[:, 0]
cal2 = cal[cal['available']=='t'].groupby(['date'])['available'].count().reset_index()
avail = pd.DataFrame(cal2['available'][:-1].values.reshape(-1, 7))
avail_week_mean = avail.mean(axis=1)
avail_week_std = avail.std(axis=1)

# figure
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)
fig = plt.figure(figsize=(40, 40))
gs = GridSpec(2, 2)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])

ax1.plot(cal1['date'], cal1['mean'], 'k')
ax1.fill_between(cal1['date'], cal1['mean']+cal1['std'], cal1['mean']-cal1['std'],
                 color='b', alpha=0.2)
ax1.set_ylabel('Average price', fontsize=20)
ax1.set_ylim([0, 300])

ax12 = ax1.twinx()
ax12.plot(cal2['date'], cal2['available'], 'r')
ax12.set_ylabel('Availability', color='r', fontsize=20)
ax12.set_ylim([0, 3000])
ax12.xaxis.set_major_locator(ticker.MultipleLocator(50))
ax12.tick_params(axis='y', colors='r')

ax2.plot(mondays, price_week_mean, 'k')
ax2.fill_between(mondays, price_week_mean-price_week_std, price_week_mean+price_week_std,
                 color='b', alpha=0.2)
ax2.tick_params(axis='x', labelrotation=10)
ax2.set_ylabel('Weekly average price', fontsize=20)
ax2.set_ylim([0, 160])

ax22 = ax2.twinx()
ax22.plot(mondays, avail_week_mean, 'r')
ax22.fill_between(mondays, avail_week_mean-avail_week_std, avail_week_mean+avail_week_std,
                 color='b', alpha=0.2)
ax22.set_ylabel('Weekly average availability', color='r', fontsize=20)
ax22.xaxis.set_major_locator(ticker.MultipleLocator(15))
ax22.set_ylim([0, 3000])
ax22.xaxis.set_major_locator(ticker.MultipleLocator(15))
ax22.tick_params(axis='y', colors='r')

price.boxplot(grid=False, ax=ax3, positions=list(range(1, 8)), widths=0.6)
xtickNames = plt.setp(ax3, xticklabels=weekdays)
plt.setp(xtickNames, fontsize=20)
ax3.set_ylabel('Weekday average price', fontsize=20)
ax3.set_ylim([0, 160])

fig.subplots_adjust(wspace=0.4)
