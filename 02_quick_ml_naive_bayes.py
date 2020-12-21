"""

This script gives a VERY basic step by step approach to use the Naive Bayes algorithm
using the scikit-learn python library. The library includes a variety of classifiers
depending on the distribution of your data. At the end of the scrip I just apply a
prediction for each possible 'weather' class. This obviously predicts the same class
for ALL gym members given the weather, which is a poor generalization, but you get
the point. Enjoy!

"""

import pandas as pd
from sklearn import preprocessing
from sklearn.naive_bayes import CategoricalNB
import numpy as np

# create some data
weather = pd.Series(['sunny', 'rainy', 'snowy', 'cloudy', 'cloudy', 'snowy', 'sunny', 'rainy',
                     'sunny', 'cloudy', 'snowy', 'sunny', 'rainy', 'cloudy', 'snowy'])

attend = pd.Series(['yes', 'no', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no', 'no', 'no', 'yes', 'no'])

# create pandas dataframe and rename columns
globo_df_long = pd.concat([weather, attend], axis=1)
globo_df_long.columns = ['weather', 'attended']

print(globo_df_long.head())

#   weather attended
# 0   sunny      yes
# 1   rainy       no
# 2   snowy       no
# 3  cloudy      yes
# 4  cloudy       no


# create frequency table
print(pd.crosstab(index=globo_df_long['weather'], columns=globo_df_long['attended']))

# attended  no  yes
# weather
# cloudy     1    3
# rainy      2    1
# snowy      3    1
# sunny      1    3


# encode the features
oe_weather = preprocessing.OrdinalEncoder()
le_attend = preprocessing.LabelEncoder()

# ordinal encoder for weather feature
oe_weather.fit(np.array(globo_df_long['weather']).reshape(-1,1))
weather = oe_weather.transform(np.array(globo_df_long['weather']).reshape(-1,1))

# label encoder for target
le_attend.fit(globo_df_long['attended'])
attend_y = le_attend.fit_transform(globo_df_long['attended'])

# view encodings
print(oe_weather.categories_)
# [array(['cloudy', 'rainy', 'snowy', 'sunny'], dtype=object)]

print(le_attend.classes_)
# ['no' 'yes']


# set and fit classifier
clf = CategoricalNB()
clf.fit(weather, attend_y)

# predict and view any given weather value
for i in range(4):
    print("weather", i, "-",
          "attendance probability:", np.round(clf.predict_proba([[i]]), 2),
          ", predicted attendance:", clf.predict([[i]])[0])


# weather 0 - attendance probability: [[0.32 0.68]] , predicted attendance: 1
# weather 1 - attendance probability: [[0.59 0.41]] , predicted attendance: 0
# weather 2 - attendance probability: [[0.66 0.34]] , predicted attendance: 0
# weather 3 - attendance probability: [[0.32 0.68]] , predicted attendance: 1
