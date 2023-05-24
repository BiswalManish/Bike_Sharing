from model import  *

### 3.3 RESIDUAL ANALYSIS

def residual_analysis(X_train, y_train, model):
    
        
    y_train_pred = model.predict(sm.add_constant(X_train))

    
    res = y_train - y_train_pred
    
    for i in X_train.columns:
    
        plt.figure(figsize = (10, 5))    
        plt.subplot(1, 2, 1)
        sns.distplot(x = res, color = 'C3')
    
        plt.subplot(1, 2, 2)
        sns.scatterplot(x = X_train[i], y = res, color = 'C2')
    

#         plt.savefig(f'Residual analysis for {i}', dpi = 500)

        plt.show() 

# residual analysis for model5

residual_analysis(X_train, y_train, Model5)


# From residual analysis we can see that the errors are normally distributed and the distribution of errors with respect to each variable is random. That is the residuals show homscadasticity. 

# In other words the variance in independent variables throughout the data domain doesnot affect the residuals' variance in the domain of the data



### 3.4 MODEL PREDICTION

df_test.head()

# Rescalling test data
df_test[num_cols] = scaler.transform(df_test[num_cols])
df_test.head()

#Creating X_test and y_test
X_test = df_test[X_train.columns]
y_test = df_test.cnt


# Prediction and evaluation
def model_eval(X_test, y_test, model):
    
    y_test_pred = model.predict(sm.add_constant(X_test))
        
    r2 = r2_score(y_true = y_test,
                  y_pred = y_test_pred)
    
    mse = mean_squared_error(y_true = y_test,
                             y_pred = y_test_pred )
    
    plt.scatter(x = y_test, y = y_test_pred, color = 'C1')
    plt.xlabel('y_test')
    plt.ylabel('y_test_pred')
    plt.title('y_test vs y_pred')
    plt.axis('square')
#     plt.savefig('y_test vs y_pred.png', dpi = 500)
    plt.show()
    
    
    print('\nCoeff of determination, r2: {}'.format(r2))
    print('\nMean squared error, mse: {}'.format(mse))
    
    return r2, mse

model_eval(X_test, y_test, Model5)

### The model seems to be working fine with test r2 score of `78.15%` which is very close to the training r2 score Also the mse is very low.





