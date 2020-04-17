
import numpy as np
import pandas as pd
import re

rev = pd.read_csv('../Data/Project_1/reviews.csv')

rev = rev[['listing_id', 'comments']]
rev.dropna(inplace=True)
rev['comments'] = rev.apply(lambda row: ' '.join(re.sub('\W', ' ', row['comments']).split()), axis=1)