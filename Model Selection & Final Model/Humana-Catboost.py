#!/usr/bin/env python
# coding: utf-8

# # Load Packages

# In[1]:


#basic libs
import pandas as pd
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np
from statistics import *
import time
import os
import pprint
import joblib
import warnings
warnings.filterwarnings("ignore")

#evaluation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score

# Model selection
from sklearn.model_selection import StratifiedKFold

#SMOTE
import imblearn
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

#CatBoost
from catboost import CatBoostClassifier
from catboost import Pool, cv

# Skopt functions
from skopt import BayesSearchCV
from skopt.callbacks import DeadlineStopper, VerboseCallback, DeltaXStopper
from skopt.space import Real, Categorical, Integer
from time import time


# In[2]:


print(os.getcwd())
os.chdir('D:/OneDrive/ASU/Humana_Case_Competition')
print(os.getcwd())


# # Read Data & Split

# In[4]:


humana = pd.read_csv('Train_Dummy.csv')
label = humana['transportation_issues']
data = humana.drop(['person_id_syn','transportation_issues'], axis = 1)
Data=data.fillna(data.mean())


# In[3]:


HUM = pd.read_csv('train.csv')
HUM.head()


# In[4]:


labels = HUM['transportation_issues']
datas = HUM.drop(['person_id_syn','transportation_issues'], axis = 1)
#Data=data.fillna(data.mean())


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(datas, labels, test_size = 0.2, random_state = 42)


# In[5]:


# transform the dataset
over = SMOTE(sampling_strategy=0.3)
under = RandomUnderSampler(sampling_strategy=0.6)
steps = [('over', over), ('under', under)]
pipeline = Pipeline(steps=steps)
X, y = pipeline.fit_resample(X_train, y_train)


# # CatBoost

# In[6]:


model=CatBoostClassifier()
model.fit(X_train,y_train,plot=True)


# In[10]:


# make predictions for test data
y_pred = model.predict_proba(X_test)
predictions = [round(value) for value in y_pred[:,1]]


# evaluate predictions
accuracy = accuracy_score(y_test, predictions)

#laucpr
print("Predict test set... ")
#test_prediction = DecisionTree.predict(X_test)
score = average_precision_score(y_test, predictions)

#auc_roc
fpr, tpr, threshold = metrics.roc_curve(y_test, predictions)
roc_auc = metrics.auc(fpr, tpr)



print("Accuracy: %.2f%%" % (accuracy * 100.0),
      'area under the precision-recall curve test set: {:.6f}'.format(score),
     "roc:",roc_auc,)


# In[18]:


cm=confusion_matrix(y_test,np.round(y_pred,0))
cm


# In[13]:


plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.02, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()


# # Tuning

# In[80]:


# Reporting util for different optimizers
def report_perf(optimizer, X, y, title, callbacks=None):
    """
    A wrapper for measuring time and performances of different optmizers
    
    optimizer = a sklearn or a skopt optimizer
    X = the training set 
    y = our target
    title = a string label for the experiment
    """
    start = time()
    if callbacks:
        optimizer.fit(X, y, callback=callbacks)
    else:
        optimizer.fit(X, y)
    d=pd.DataFrame(optimizer.cv_results_)
    best_score = optimizer.best_score_
    best_score_std = d.iloc[optimizer.best_index_].std_test_score
    best_params = optimizer.best_params_
    print((title + " took %.2f seconds,  candidates checked: %d, best CV score: %.3f "
           +u"\u00B1"+" %.3f") % (time() - start, 
                                  len(optimizer.cv_results_['params']),
                                  best_score,
                                  best_score_std))    
    print('Best parameters:')
    pprint.pprint(best_params)
    print()
    return best_params


# In[81]:


roc_auc = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


# In[83]:


clf = CatBoostClassifier(thread_count=2,
                         loss_function='Logloss',
                         od_type = 'Iter',
                         verbose= False
                        )


# In[84]:


# Defining your search space
search_spaces = {'iterations': Integer(300, 1000),
                 'depth': Integer(4, 12),
                 'learning_rate': Real(0.005, 1.0, 'log-uniform'),
                 'random_strength': Real(1e-9, 10, 'log-uniform'),
                 'bagging_temperature': Real(0.0, 1.0),
                 'border_count': Integer(1, 255),
                 'l2_leaf_reg': Integer(2, 30),
                 'scale_pos_weight':Real(0.5, 3.0, 'uniform')}


# In[85]:


# Setting up BayesSearchCV
opt = BayesSearchCV(clf,
                    search_spaces,
                    scoring=roc_auc,
                    cv=skf,
                    n_iter=100,
                    n_jobs=1,  # use just 1 job with CatBoost in order to avoid segmentation fault
                    return_train_score=False,
                    refit=True,
                    optimizer_kwargs={'base_estimator': 'GP'},
                    random_state=42)


# In[86]:


best_params = report_perf(opt, X_train, y_train,'CatBoost', 
                           callbacks=[VerboseCallback(100), 
                                      DeadlineStopper(60*10)])


# In[22]:


class ModelOptimizer:
    best_score = None
    opt = None
    
    def __init__(self, model, X_train, y_train, categorical_columns_indices=None, n_fold=3, seed=2405, early_stopping_rounds=30, is_stratified=True, is_shuffle=True):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.categorical_columns_indices = categorical_columns_indices
        self.n_fold = n_fold
        self.seed = seed
        self.early_stopping_rounds = early_stopping_rounds
        self.is_stratified = is_stratified
        self.is_shuffle = is_shuffle
        
        
    def update_model(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self.model, k, v)
            
    def evaluate_model(self):
        pass
    
    def optimize(self, param_space, max_evals=10, n_random_starts=2):
        start_time = time.time()
        
        @use_named_args(param_space)
        def _minimize(**params):
            self.model.set_params(**params)
            return self.evaluate_model()
        
        opt = gp_minimize(_minimize, param_space, n_calls=max_evals, n_random_starts=n_random_starts, random_state=2405, n_jobs=-1)
        best_values = opt.x
        optimal_values = dict(zip([param.name for param in param_space], best_values))
        best_score = opt.fun
        self.best_score = best_score
        self.opt = opt
        
        print('optimal_parameters: {}\noptimal score: {}\noptimization time: {}'.format(optimal_values, best_score, time.time() - start_time))
        print('updating model with optimal values')
        self.update_model(**optimal_values)
        plot_convergence(opt)
        return optimal_values
    
class CatboostOptimizer(ModelOptimizer):
    def evaluate_model(self):
        validation_scores = catboost.cv(
        catboost.Pool(self.X_train, 
                      self.y_train, 
                      cat_features=self.categorical_columns_indices),
        self.model.get_params(), 
        nfold=self.n_fold,
        stratified=self.is_stratified,
        seed=self.seed,
        early_stopping_rounds=self.early_stopping_rounds,
        shuffle=self.is_shuffle,
        metrics='auc',
        plot=False)
        self.scores = validation_scores
        test_scores = validation_scores.iloc[:, 2]
        best_metric = test_scores.max()
        return 1 - best_metric


# In[32]:


cb = CatBoostClassifier(n_estimators=4000, # use large n_estimators deliberately to make use of the early stopping
                         one_hot_max_size=2,
                         loss_function='Logloss',
                         eval_metric='AUC',
                         boosting_type='Ordered', # use permutations
                         random_seed=2405, 
                         use_best_model=True,
                         silent=True)
one_cb_optimizer = CatboostOptimizer(cb, X_train, y_train)
params_space = [Real(0.01, 0.8, name='learning_rate'), 
                Integer(2, 16, name='max_depth'), 
                Real(0.5, 1.0, name='colsample_bylevel'), 
                Real(1.0, 16.0, name='scale_pos_weight'), 
                Real(0.0, 100, name='bagging_temperature'), 
                Real(0.0, 100, name='random_strength'), 
                Real(1.0, 100, name='reg_lambda')]
one_cb_optimal_values = one_cb_optimizer.optimize(params_space, max_evals=40, n_random_starts=4)


# In[8]:


best_params={
    'bagging_temperature': 0.41010395885331385,
    'border_count': 186,
    'depth': 11,
    'iterations': 1000,
    'l2_leaf_reg': 21,
    'learning_rate': 0.044861046400920826,
    'random_strength': 3.230824361824754e-06,
    'loss_function':'Logloss',
    'eval_metric':'AUC',
    'scale_pos_weight': 2.348760585476051,
#    'class_weights':[0.5,2],
    'custom_metric':['Logloss', 'AUC']
    
            }


# In[14]:


get_ipython().run_cell_magic('time', '', 'tuned_model = CatBoostClassifier(**best_params,task_type = "GPU",od_type=\'Iter\',early_stopping_rounds=30)\ntuned_model.fit(X_train,y_train,plot=True)')


# In[29]:


# make predictions for test data
y_pred = tuned_model.predict_proba(X_test)
predictions = [round(value) for value in y_pred[:,1]]


# evaluate predictions
accuracy = accuracy_score(y_test, predictions)

#laucpr
print("Predict test set... ")
#test_prediction = DecisionTree.predict(X_test)
score = average_precision_score(y_test, predictions)

#auc_roc
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred[:,1])
roc_auc = metrics.auc(fpr, tpr)



print("Accuracy: %.2f%%" % (accuracy * 100.0),
      'area under the precision-recall curve test set: {:.6f}'.format(score),
     "roc:",roc_auc,)


# In[91]:


cm=confusion_matrix(y_test,np.round(y_pred,0))
cm


# In[74]:


plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.02, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()


# In[ ]:




