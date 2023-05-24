from data import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

from sklearn.feature_selection import RFE
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score, mean_squared_error


# Dropping unnecessary cols
df = day.drop(['instant', 'dteday', 'casual', 'registered'], axis = 1)
df.head()

# Creating dummy variable
df_season = pd.get_dummies(df.season).rename({1 : 'spring', 2 : 'summer',
                                              3 : 'fall', 4 :  'winter'}, axis = 1).drop(['spring'], axis =1)


df_mnth = pd.get_dummies(df.mnth).rename({1 : 'Jan', 2 : 'Feb', 3 : 'Mar', 4 : 'Apr',
                                            5 : 'May', 6 : 'Jun', 7 : 'Jul', 8 : 'Aug',
                                            9 : 'Sep', 10 : 'Oct', 11 : 'Nov', 12 : 'Dec'},
                                           axis = 1).drop(['Jan'], axis =1)


df_weekday = pd.get_dummies(df.weekday).rename({0 : 'Tue', 1 : 'Wed', 2: 'Thu', 3: 'Fri',
                                                4 : 'Sat', 5 : 'Sun', 6 : 'Mon'}, 
                                              axis = 1).drop(['Tue'], axis = 1)


df_weather = pd.get_dummies(df.weathersit).rename({1 : 'Clear', 2 : 'Clody_misty',
                                              3 : 'Rainy_snow'}, axis = 1).drop(['Clear'], axis =1)



# Concatanating dummy df
df = pd.concat([df, df_season, df_mnth, df_weekday, df_weather], axis = 1).drop(['season',
                                                                'mnth', 'weekday', 'weathersit'],
                                                               axis = 1)



# Train test split

df_train, df_test = train_test_split(df, test_size = 0.3,
                                     random_state = 42)


# Rescalling

num_cols.remove('casual')
num_cols.remove('registered')

# rescalling numerical columns
scaler = MinMaxScaler()

df_train[num_cols] = scaler.fit_transform(df_train[num_cols])


# Creating X and y

X_train, y_train = df_train.drop(['cnt'], axis = 1), df_train['cnt']


# Correlation coeff

corr = df_train.corr()

# plt.figure(figsize = (25,25))
# sns.heatmap(corr, annot = True, cmap = 'Greens')
# plt.savefig('Heatmap', dpi = 500)
# plt.show()

# creating a list of sorted variables as per their corr with cnt
var = corr.cnt.sort_values()


