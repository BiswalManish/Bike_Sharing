from model_prep import *

# Creating variable sorted with their abs values
var = pd.DataFrame(data = var)
var['abs_values'] = var.apply(lambda x : abs(x))

var = list(var['abs_values'].sort_values(ascending = False).index)
var.remove('cnt')

# Building a linear model
def build_linear_model(X, y):
    
    model = sm.OLS(y, sm.add_constant(X))
    model = model.fit()
#     print(model.rsquared_adj)
    
    return model

# Checking VIF scores for independent variables
def get_vif(X_train):
    vif_df = pd.DataFrame()

    vif_df['Features'] = X_train.columns
    vif_df['VIF'] = [variance_inflation_factor(X_train.values, i)
                 for i in range(X_train.shape[1])]

    vif_df = vif_df.sort_values(by = 'VIF', ascending = False)
    return vif_df



#Models

# Model 0

model0 = build_linear_model(X_train, y_train)
print(model0.summary2())
print('\n\nAdj_r2 for model 0: ',model0.rsquared_adj)


# Checking adjusted r2
var_list = []
for i in range(1,len(var) + 1):
    k = var[ : i]
    var_list.append(k)

Adj_r2 = []
for i in var_list:
    model = build_linear_model(X_train[i], y_train)
    k = round(100 * (model.rsquared_adj), 3)
    Adj_r2.append(k)
    
plt.figure(figsize = (8,8))
plt.plot(Adj_r2, color = 'C3')
plt.xlabel('Features')
plt.ylabel('Adjusted_r2')
# plt.savefig('Adjusted_r2 vs features', dpi = 500)
plt.show()


get_vif(X_train)



# Feature Elimination

lr = LinearRegression()
lr = lr.fit(X_train, y_train)

rfe = RFE(estimator = lr,
          n_features_to_select = 15)

rfe = rfe.fit(X_train, y_train)

RFE_ = pd.DataFrame(list(zip(X_train.columns, rfe.support_, rfe.ranking_)),
                    columns = ['Features', 'Support', 'Ranking'])

print('\n'*4)
print(RFE_)

##### Eliminating features not supported by RFE
sup = X_train.columns[rfe.support_]
rej = X_train.columns[~rfe.support_]

#Creating rfe supported X_train
X_train = X_train[sup]


# Model 1

Model1 = build_linear_model(X_train, y_train)
print(Model1.summary2())
print('\n\n')
print(get_vif(X_train))
print('\n\n')
#### Model 2

X_train = X_train.drop(['Thu', 'Fri', 'Sat', 'Sun'], axis = 1)

Model2 = build_linear_model(X_train, y_train)
print(Model2.summary2())
print('\n\n')
print(get_vif(X_train))
print('\n\n')


#### Model 3

X_train = X_train.drop(['workingday', 'holiday', 'Wed'], axis = 1)

Model3 = build_linear_model(X_train, y_train)
print(Model3.summary2())
print('\n\n')
print(get_vif(X_train))
print('\n\n')


#### Model 4

Model4 = build_linear_model(X_train.drop(['fall'], axis = 1), y_train)
print(Model4.summary2())
print('\n\n')
print(get_vif(X_train.drop(['fall'], axis = 1)))
print('\n\n')


#### Model 5

Model5 = build_linear_model(X_train.drop(['fall', 'windspeed'], axis = 1), y_train)
print(Model5.summary2())
print('\n\n')
print(get_vif(X_train.drop(['fall', 'windspeed'], axis = 1)))
print('\n\n')

#### Saving X_train

X_train = X_train.drop(['fall', 'windspeed'], axis = 1)

#### Saving Models

models = [model0, Model1, Model2,
          Model3, Model4, Model5]

# for model in range(len(models)):
#     fname = f'Model{model}'
#     pickle.dump(fname, open(fname, 'wb'))

