
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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

# rent length
rent_length = pd.Series([], dtype='float')
for i in occupancy.columns:
    a = occupancy[i].eq(1)
    b = a.cumsum()
    c = b.sub(b.mask(a).ffill().fillna(0)).astype(int)
    d = c[c>c.shift(-1)]
    rent_length = rent_length.add(d.value_counts(), fill_value=0)
rent_length2 = rent_length[:4]
rent_length2['5+'] = rent_length[4:].sum()

# figure
fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(1, 1, 1)
(rent_length2/rent_length2.sum()*100).plot.bar(rot=0, ax=ax1)
plt.tick_params(labelsize=15)
ax1.set_xlabel('Stay length (days)', fontsize=20)
ax1.set_ylabel('Relative count (%)', fontsize=20)
plt.savefig('Figures/Figure 7.png', bbox_inches='tight')
