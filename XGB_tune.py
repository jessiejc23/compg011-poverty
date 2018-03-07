
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


tr_A_h = pd.read_csv('A_hhold_train.csv',index_col='id')
tr_B_h = pd.read_csv('B_hhold_train.csv',index_col='id')
tr_C_h = pd.read_csv('C_hhold_train.csv',index_col='id')


# In[3]:


ts_A_h = pd.read_csv('A_hhold_test.csv',index_col='id')
ts_B_h = pd.read_csv('B_hhold_test.csv',index_col='id')
ts_C_h = pd.read_csv('C_hhold_test.csv',index_col='id')


# In[4]:


print('tr_A\n',tr_A_h.info(),'\n',tr_A_h.isnull().any().sum(),'\n==================')
print('tr_B\n',tr_B_h.info(),'\n',tr_B_h.isnull().any().sum(),'\n==================')
print('tr_C\n',tr_C_h.info(),'\n',tr_C_h.isnull().any().sum(),'\n==================')
print('ts_A\n',ts_A_h.info(),'\n',ts_A_h.isnull().any().sum(),'\n==================')
print('ts_B\n',ts_B_h.info(),'\n',ts_B_h.isnull().any().sum(),'\n==================')
print('ts_C\n',ts_C_h.info(),'\n',ts_C_h.isnull().any().sum(),'\n==================')


# In[5]:


# Standardize features
def standardize(df, numeric_only=True):
    numeric = df.select_dtypes(include=['int64', 'float64'])
    # subtracy mean and divide by std
    df[numeric.columns] = (numeric - numeric.mean()) / numeric.std()
    
    return df
    
def pre_process_data(df, enforce_cols=None):
    print("Input shape:\t{}".format(df.shape))
    df = standardize(df)
    print("After standardization {}".format(df.shape))
        
    # create dummy variables for categoricals
    df = pd.get_dummies(df)
    print("After converting categoricals:\t{}".format(df.shape))
    
    '''
    # match test set and training set columns
    if enforce_cols is not None:
        to_drop = np.setdiff1d(df.columns, enforce_cols)
        to_add = np.setdiff1d(enforce_cols, df.columns)

        df.drop(to_drop, axis=1, inplace=True)
        df = df.assign(**{c: 0 for c in to_add})
    
    df.fillna(0, inplace=True)
    '''
    return df


# In[6]:


print("Country A")
X_train_A= pre_process_data(tr_A_h.drop('poor', axis=1))
y_train_A= np.ravel(tr_A_h.poor)
print('test')
X_test_A= pre_process_data(ts_A_h)

print("\nCountry B")
X_train_B= pre_process_data(tr_B_h.drop('poor', axis=1))
y_train_B= np.ravel(tr_B_h.poor)
print('test')
X_test_B= pre_process_data(ts_B_h)

print("\nCountry C")
X_train_C= pre_process_data(tr_C_h.drop('poor', axis=1))
y_train_C= np.ravel(tr_C_h.poor)

print('test')
X_test_C= pre_process_data(ts_C_h)


c_trA = X_train_A.columns.tolist()
c_trB = X_train_B.columns.tolist()
c_trC = X_train_C.columns.tolist()
c_tsA = X_test_A.columns.tolist()
c_tsB = X_test_B.columns.tolist()
c_tsC = X_test_C.columns.tolist()


# In[7]:


dif_A = list(set(c_trA).difference(set(c_tsA)))
dif_B = list(set(c_trB).difference(set(c_tsB)))
dif_C = list(set(c_trC).difference(set(c_tsC)))


# In[8]:


X_train_A = X_train_A.drop(dif_A,axis=1)
X_train_B = X_train_B.drop(dif_A,axis=1)
X_train_C = X_train_C.drop(dif_A,axis=1)


# In[9]:


c_trA = X_train_A.columns.tolist()
c_trB = X_train_B.columns.tolist()
c_trC = X_train_C.columns.tolist()
c_tsA = X_test_A.columns.tolist()
c_tsB = X_test_B.columns.tolist()
c_tsC = X_test_C.columns.tolist()


# In[10]:


dif_a = list(set(c_tsA).difference(set(c_trA)))
dif_b = list(set(c_tsB).difference(set(c_trB)))
dif_c = list(set(c_tsC).difference(set(c_trC)))

X_test_B = X_test_B.drop(dif_b,axis = 1)
X_test_C = X_test_C.drop(dif_c,axis = 1)


# In[11]:


print('tr_A\n',X_train_A.info(),'\n',X_train_A.isnull().any().sum(),'\n==================')
print('tr_B\n',X_train_B.info(),'\n',X_train_B.isnull().any().sum(),'\n==================')
print('tr_C\n',X_train_C.info(),'\n',X_train_C.isnull().any().sum(),'\n==================')


# In[12]:


#Preprocessing End


# In[ ]:


#TUNE THE MODEL


# In[13]:


from sklearn.model_selection import train_test_split, KFold
#from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score


# In[9]:



#X_tr_A,X_ts_A, y_tr_A,y_ts_A = train_test_split(X_train_A,y_train_A,test_size = 0.1,random_state = 0)
#X_tr_B,X_ts_B, y_tr_B,y_ts_B = train_test_split(X_train_B,y_train_B,test_size = 0.1,random_state = 0)
#X_tr_C,X_ts_C, y_tr_C,y_ts_C = train_test_split(X_train_C,y_train_C,test_size = 0.1,random_state = 0)


# In[17]:


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


# In[9]:


#tune 1:
depth = [3,4]
learning = [0.001,0.05,0.1,0.15,0.2,0.3]
estimators = [100,150,200,300]


#########
model = XGBClassifier(random_state =0)
param_grid = dict(max_depth = depth,learning_rate=learning, n_estimators=estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold,verbose=True)
grid_result = grid_search.fit(X_trian_A, y_train_A)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[ ]:


#Model_A_XgB
#Best: -0.207949 using 
#{'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 150}


# In[11]:


#Tune 2
gamma = [0,0.1,0.2,0.3,0.4,0.5]
min_child_weight =[1,3,5,7,9]

#######
model = XGBClassifier(max_depth=3,n_estimators=150,learning_rate=0.1)#tune 1 result

param_grid = dict(gamma = gamma,min_child_weight=min_child_weight)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold,verbose=True)
grid_result = grid_search.fit(X_trian_A, y_train_A)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[ ]:


#Best: -0.207599 using 
#{'gamma': 0.1, 'min_child_weight': 1}


# In[27]:


#Tune 3
subsample = [0.85,0.9,0.95]
colsample_bytree=[0.55,0.6,0.65]



model = XGBClassifier(max_depth=3,
                      n_estimators=150,
                      learning_rate=0.1,
                      gamma = 0.1,
                      min_child_weight =1,
                      random_state=0)       #tune 2 result

param_grid = dict(subsample = subsample,colsample_bytree=colsample_bytree)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold,verbose=True)
grid_result = grid_search.fit(X_trian_A, y_train_A)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[11]:


#Best: -0.206800 using 
#{'colsample_bytree': 0.6, 'subsample': 0.9}

#Best: -0.206398 using 
#{'colsample_bytree': 0.55, 'subsample': 0.9}


# In[29]:



#Tune4
reg_alpha = [0.03,0.04,0.05,0.06,0.07]

model = XGBClassifier(max_depth=3,
                      n_estimators=150,
                      learning_rate=0.1,
                      gamma = 0.1,
                      min_child_weight =1,
                      random_state=0,
                      colsample_bytree=0.55,
                      subsample=0.9
                        )

param_grid = dict(reg_alpha=reg_alpha)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold,verbose=True)
grid_result = grid_search.fit(X_trian_A, y_train_A)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[ ]:


#Best: -0.204953 using {'reg_alpha': 0.05}


# In[36]:


#Final CV test

model_A= XGBClassifier(max_depth=3,
                      n_estimators=150,
                      learning_rate=0.1,
                      gamma = 0.1,
                      min_child_weight =1,
                      random_state=0,
                      colsample_bytree=0.55,
                      subsample=0.9,
                      reg_alpha = 0.05
                        )

test_score_A = cross_val_score(model_A, X_train_A, y_train_A, cv=10, scoring='neg_log_loss')
print('A:',np.mean(test_score_A))


# In[ ]:


#After tuning ALL the model (A,B,C) should repeat the tuning process 3 times


# In[18]:


model_A= XGBClassifier(max_depth=3,
                      n_estimators=150,
                      learning_rate=0.1,
                      gamma = 0.1,
                      min_child_weight =1,
                      random_state=0,
                      colsample_bytree=0.55,
                      subsample=0.9,
                      reg_alpha = 0.05
                        )
model_B= XGBClassifier(max_depth=3,
                      n_estimators=150,
                      learning_rate=0.1,
                      gamma = 0.1,
                      min_child_weight =1,
                      random_state=0,
                      colsample_bytree=0.55,
                      subsample=0.9,
                      reg_alpha = 0.05
                        )
model_C = XGBClassifier(max_depth=4,
                      n_estimators=200,
                      learning_rate=0.05,
                      gamma = 0.5,
                      min_child_weight =1,
                      random_state=0,
                      colsample_bytree=1,
                      subsample=1)


# In[20]:


model_A.fit(X_train_A, y_train_A)
model_B.fit(X_train_B, y_train_B)
model_C.fit(X_train_C, y_train_C)


# In[21]:


a_preds = model_A.predict_proba(X_test_A)
b_preds = model_B.predict_proba(X_test_B)
c_preds = model_C.predict_proba(X_test_C)


# In[ ]:


def make_country_sub(preds, test_feat, country):
    # make sure we code the country correctly
    country_codes = ['A', 'B', 'C']
    
    # get just the poor probabilities
    country_sub = pd.DataFrame(data=preds[:, 1],  # proba p=1
                               columns=['poor'], 
                               index=test_feat.index)

    
    # add the country code for joining later
    country_sub["country"] = country
    return country_sub[["country", "poor"]]



a_sub = make_country_sub(a_preds, X_test_A, 'A')
b_sub = make_country_sub(b_preds, X_test_B, 'B')
c_sub = make_country_sub(c_preds, X_test_C, 'C')
submission = pd.concat([a_sub, b_sub, c_sub])


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv('submission.csv')

