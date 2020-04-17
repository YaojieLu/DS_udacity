
import numpy as np
import pandas as pd
import re
from textblob import TextBlob 
import matplotlib.pyplot as plt

# import csv
cal = pd.read_csv('../Data/Project_1/calendar.csv')
rev = pd.read_csv('../Data/Project_1/reviews.csv')

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
# sentiment analysis
def sentiment(text): 
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0: 
        return 'positive'
    elif analysis.sentiment.polarity == 0: 
        return 'neutral'
    else: 
        return 'negative'
rev = rev[['listing_id', 'comments']]
rev.dropna(inplace=True)
rev['comments'] = rev.apply(lambda row: ' '.join(re.sub('\W', ' ', row['comments']).split()), axis=1)
rev['sentiments'] = rev['comments'].apply(sentiment)
rev2 = rev.groupby(['listing_id'])['sentiments'].value_counts().unstack().fillna(0).reset_index()
rev2['review_count'] = rev2[['negative', 'neutral', 'positive']].sum(axis=1)

# merge
df = pd.merge(df, rev2, on='listing_id')

# figure
fig = plt.figure(figsize=(10, 10))
# review_count vs. income
ax1 = fig.add_subplot(2, 2, 1)
df.plot.scatter('review_count', 'total_income', ax=ax1)
# review_count vs. occupancy
ax2 = fig.add_subplot(2, 2, 2)
df.plot.scatter('review_count', 'total_occupancy', ax=ax2)
# review_count vs. income
ax3 = fig.add_subplot(2, 2, 3)
ax3.hist(df[df['negative']==0]['total_income'], bins=10, alpha=0.5, density=True, label=0)
ax3.hist(df[df['negative']==1]['total_income'], bins=10, alpha=0.5, density=True, label=1)
ax3.hist(df[df['negative']>1]['total_income'], bins=10, alpha=0.5, density=True, label='more')
ax3.set_xlabel('Income', fontsize=20)
ax3.tick_params(labelsize=15)
# review_count vs. occupancy
ax4 = fig.add_subplot(2, 2, 4)
ax4.hist(df[df['negative']==0]['total_occupancy'], bins=10, alpha=0.5, density=True, label=0)
ax4.hist(df[df['negative']==1]['total_occupancy'], bins=10, alpha=0.5, density=True, label=1)
ax4.hist(df[df['negative']>1]['total_occupancy'], bins=10, alpha=0.5, density=True, label='more')
ax4.legend(loc='upper right', fontsize=20)
ax4.set_xlabel('Occupancy', fontsize=20)
ax4.tick_params(labelsize=15)
