
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# import csv
cal = pd.read_csv('../Data/Project_1/calendar.csv')

# analysis
price = cal.apply(lambda row: float(re.sub('(\$|,)', '', row['price'])) if isinstance(row['price'], str) else np.nan, axis=1)
price = pd.DataFrame(price.values.reshape(365, -1))
price = price.ffill().bfill()
price_qt = price.quantile([.05, .5, 0.95], axis=1).T
price_top = ((price.T*-1).apply(lambda x: x.sort_values().values)*-1).T.iloc[:, :380]
price_top_qt = price_top.quantile([.1, .5, 0.9], axis=1).T
occupancy = pd.DataFrame(cal['available'].values.reshape(365, -1))
occupancy = occupancy.replace(['t', 'f'], [0, 1])
income = price*occupancy
income_qt = income.replace(0, np.nan).quantile([0.05, 0.5, 0.95], axis=1).T
date = cal['date'][:365]

# figure
# price
fig = plt.figure(figsize=(10, 9))
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(price_qt[0.50], color='k')
ax1.fill_between(date, price_qt[0.05], price_qt[0.95],
                 color='b', alpha=0.2)
ax1.set_xlim([0, 365])
ax1.set_ylim([0, 1100])
ax1.set_ylabel('Listing price', fontsize=20)
ax1.tick_params(labelsize=20)
ax1.axes.get_xaxis().set_visible(False)
# median daily income
ax12 = ax1.twinx()
ax12.plot(income_qt[0.50], color='r')
ax12.fill_between(date, income_qt[0.05], income_qt[0.95],
                 color='b', alpha=0.2)
ax12.set_ylim([0, 1100])
ax12.set_ylabel('Median income', color='r', fontsize=20)
ax12.tick_params(axis='y', colors='r', labelsize=20)
# total daily income
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(date, income.sum(axis=1), color='k')
ax2.set_xlim([0, 365])
ax2.set_ylim([0, 700000])
ax2.set_ylabel('Total income', fontsize=20)
ax2.tick_params(labelsize=20)
ax2.xaxis.set_major_locator(ticker.MultipleLocator(50))
ax2.tick_params(axis='x', labelrotation=20)
# occupancy
ax22 = ax2.twinx()
ax22.plot(occupancy.mean(axis=1), color='r')
ax22.set_ylim([0, 1])
ax22.set_ylabel('Mean occupancy rate', color='r', fontsize=20)
ax22.tick_params(axis='y', colors='r', labelsize=20)
plt.subplots_adjust(hspace=0.15)
plt.savefig('Figures/Figure 2.png', bbox_inches='tight')
