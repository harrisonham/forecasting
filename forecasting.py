# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 20:33:32 2024

@author: Harrison Ham
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from skopt.space import Real, Integer
from skopt import BayesSearchCV
from time import time,sleep
import os
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.decomposition import PCA
from joblib import cpu_count
import socket
from sklearn.model_selection import ParameterGrid

"""
See forecasting documentation pdf for details on this class. Email 
harrisondham@gmail.com if you have any questions.
"""

class forecasting():
    def __init__(self, 
                 data=None,
                 algorithm='GBRT',
                 objective='mae',
                 cv_scheme='timeseries',
                 refit_frequency=12,
                 parameter_frequency=-1,
                 training_window=120,
                 trim_y_var=False,
                 identifiers=['permno','gvkey'],
                 forecast_dates='tm',
                 actual_dates='announcement_month',
                 target='y_var',
                 parameter_directory='',
                 prediction_directory='',
                 result_directory='',
                 predictors=None,
                 drop_variables=[],
                 n_folds=5,
                 first_prediction_period=365,
                 verbose=True,
                 pca_data=None,
                 n_pcs=0,
                 exog_vars_NN=[],
                 cv_gap=False,
                 x_transform='std',
                 fit_kwargs={},
                 add_constant=True,
                 ):
        
        self.cv_scheme=cv_scheme
        self.forecast_dates=forecast_dates
        self.actual_dates=actual_dates
        self.n_folds=n_folds
        self.objective=objective
        self.algorithm=algorithm
        self.target=target
        self.first_prediction_period=first_prediction_period
        # self.array=array
        self.refit_frequency=refit_frequency
        self.parameter_frequency=parameter_frequency
        self.training_window=training_window
        self.predictors=predictors
        self.identifiers=identifiers
        self.data=data
        self.best_params=pd.DataFrame()
        self.trim_y_var=trim_y_var
        self.parameter_directory=parameter_directory
        self.prediction_directory=prediction_directory
        self.result_directory=result_directory
        self.fit_kwargs=fit_kwargs
        self.missing_pred_arrays=None
        self.verbose=verbose
        self.df_pca=pca_data
        self.n_pcs=n_pcs
        self.exog_vars_NN=exog_vars_NN
        self.cv_gap=cv_gap
        self.x_transform=x_transform
        self.feature_importance=None
        self.add_constant=add_constant
        
        #make directories if necessary
        # for i in [parameter_directory,prediction_directory,result_directory]:
        def mkdirs(i):
            slash=None
            if '/' in i:
                slash='/'
            elif r'\\' in i:
                slash=r'\\'
            
            if not slash==None :
                file_add=i[-i[::-1].find(slash):]
                folder=i.replace(file_add,'') if file_add!=i else i[:-1]
                if not os.path.isdir(folder):
                    os.mkdir(folder)
            else:
                file_add=i
                folder=''
            
            return file_add, folder
        
        
        
        #make directories if they dont exist
        param_file_start,self.parameter_directory=mkdirs(parameter_directory)
        pred_file_start,self.prediction_directory=mkdirs(prediction_directory)
        result_file_start,self.result_directory=mkdirs(result_directory)
        self.param_file_name=param_file_start+algorithm+'_'+objective+'_'+cv_scheme+'_'+str(training_window)+'_'+str(trim_y_var)
        self.pred_file_name=pred_file_start+algorithm+'_'+objective+'_'+cv_scheme+'_'+str(training_window)+str(trim_y_var)+'_'+str(refit_frequency)+'_'+str(parameter_frequency)
        self.result_file_name=result_file_start+algorithm+'_'+objective+'_'+cv_scheme+'_'+str(training_window)+str(trim_y_var)+'_'+str(refit_frequency)+'_'+str(parameter_frequency)
        
        #Sort the data
        self.data=self.data.sort_values(by=[self.forecast_dates]).reset_index(drop=True)
        
        #Variables we dont want in the training/test input datasets
        self.identifiers.extend([self.forecast_dates,self.actual_dates])
        self.data=self.data.drop(columns=drop_variables)
        
        
        all_arrays=[x for x in sorted(self.data[self.forecast_dates].unique()) if x>=self.first_prediction_period]
        
        #get current forecast date based on the array number
        if self.refit_frequency<=0:
            self.pred_arrays=[first_prediction_period]
        else:
            self.pred_arrays=[all_arrays[self.refit_frequency*x] for x in np.arange(len(all_arrays)//self.refit_frequency)]
        
        self.n_pred_arrays=len(self.pred_arrays)
        if verbose:
            print('Need to run '+str(self.n_pred_arrays)+' prediction array(s).')
        
        #get parameter generation date based on the array number
        if not algorithm in ['OLS','esLGBM','NHN']:
            if self.parameter_frequency<=0:
                self.param_arrays=[first_prediction_period]
            else:
                self.param_arrays=[all_arrays[self.parameter_frequency*x] for x in np.arange(len(all_arrays)//self.parameter_frequency)]
            
            self.n_param_arrays=len(self.param_arrays)
            if verbose:
               print('Need to run '+str(self.n_param_arrays)+' parameter array(s).')
        else:
            self.n_param_arrays=0
        
    def data_clean(self,date):
        
        #create testing set
        try:
            self.X_test=self.data[np.logical_and(self.data[self.forecast_dates]>=date,self.data[self.forecast_dates]<self.pred_arrays[self.pred_arrays.index(date)+1])]
        except IndexError:
            self.X_test=self.data[self.data[self.forecast_dates]>=date]
        
        
        #create training set
        if self.training_window<=0:
            self.X_train=self.data[self.data[self.actual_dates]<=date]
        else:
            self.X_train=self.data[self.data[self.actual_dates].between(date-self.training_window+1,date)]
        
        self.gap=self.X_test[self.forecast_dates].min()-self.X_train[self.forecast_dates].max()
        
        #Trim self.target in train set only if trim_y_var==True
        if self.trim_y_var:
            self.X_train[self.target]=np.where(self.X_train[self.target].quantile(.01)>self.X_train[self.target],np.nan,np.where(self.X_train[self.target].quantile(.99)<self.X_train[self.target] ,np.nan,self.X_train[self.target]))
        
        #Drop missing y variables in training set
        self.X_train=self.X_train.dropna(subset=[self.target]).reset_index(drop=True)
        
        #Make the cross validation indexes
        self.CV_Scheme(no_overlap=self.cv_gap)
        
        #Make training and test target dataframes.
        self.Y_train=self.X_train[self.identifiers+[self.target]]
        self.Y_test=self.X_test[self.identifiers+[self.target]]
        
        if self.n_pcs>0:
            self.pca_predictors()
        
        #Drop index variables
        if self.predictors:
            self.X_train=self.X_train[self.predictors]
            self.X_test=self.X_test[self.predictors]
        else:
            self.X_train=self.X_train.drop(columns=self.identifiers+[self.target])
            self.X_test=self.X_test.drop(columns=self.identifiers+[self.target])
        
        #drop completely missing predictors
        self.X_test=self.X_test.drop(self.X_train.columns[self.X_train.isna().mean()==1],axis=1)
        self.X_train=self.X_train.drop(self.X_train.columns[self.X_train.isna().mean()==1],axis=1)
        
        #Standardize
        if self.x_transform=='std':
            self.X_test=(self.X_test-self.X_train.mean())/self.X_train.std()
            self.X_train=(self.X_train-self.X_train.mean())/self.X_train.std()
        elif self.x_transform=='minmax':
            self.X_test = ((self.X_test - self.X_train.min(axis=0)) / (self.X_train.max(axis=0) - self.X_train.min(axis=0)))*(2)-1
            self.X_test =pd.DataFrame(np.where(self.X_test>1,1,np.where(self.X_test<-1,-1,self.X_test)),index=self.X_test.index,columns=self.X_test.columns)
            self.X_train = ((self.X_train - self.X_train.min(axis=0)) / (self.X_train.max(axis=0) -self. X_train.min(axis=0)))*(2)-1
        
        self.X_train=self.X_train.fillna(0)
        self.X_test=self.X_test.fillna(0)
        
        #Add constant
        if self.add_constant:
            self.X_train['const']=1
            self.X_test['const']=1
        
    
    def CV_Scheme(self,no_overlap=False):
        index_output=[]
        if self.cv_scheme=='kfold':
            index_scrambled=np.random.choice(self.X_train.index,self.X_train.shape[0],replace=False)
            
            for ifold in np.arange(self.n_folds):
                #index_output list
                train=index_scrambled[int((ifold)*self.X_train.shape[0]/self.n_folds):int((ifold+1)*self.X_train.shape[0]/self.n_folds)]
                index_output.append([train,
                                    self.X_train.index[~self.X_train.index.isin(train)]])
            
        if self.cv_scheme=='timeseries':
            #get list of months
            tms=pd.DataFrame(self.X_train[self.forecast_dates].unique())
            
            
            tms['fold'] = pd.qcut(tms[0].rank(method='first'), self.n_folds, labels=False)
            
            
            index_output.append([self.X_train.index[~self.X_train[self.forecast_dates].isin(tms.loc[tms.fold==self.n_folds-1][0])],
                                 self.X_train.index[self.X_train[self.forecast_dates].isin(tms.loc[tms.fold==self.n_folds-1][0])]])
            if no_overlap:
                index_output[0][0]=index_output[0][0][:(len(index_output[0][0])-self.gap)]
                
        if self.cv_scheme=='temporal':
            #get list of months
            tms=pd.DataFrame(self.X_train[self.forecast_dates].unique())
            
            
            tms['fold'] = pd.qcut(tms[0].rank(method='first'), self.n_folds, labels=False)
            
            tms=tms.sort_values(by=[0])
            
            for i in np.arange(self.n_folds):
                max_loc=list(tms[0]).index(tms.loc[tms.fold==i][0].max())
                min_loc=list(tms[0]).index(tms.loc[tms.fold==i][0].min())
                if no_overlap:
                    index_output.append([self.X_train.index[~self.X_train[self.forecast_dates].isin(tms.loc[(tms.index>=(min_loc-self.gap))*(tms.index<(max_loc+self.gap))][0])],
                                 self.X_train.index[self.X_train[self.forecast_dates].isin(tms.loc[(tms.index>=min_loc)*(tms.index<=max_loc)][0])]])
                else:
                    index_output.append([self.X_train.index[~self.X_train[self.forecast_dates].isin(tms.loc[(tms.index>=min_loc)*(tms.index<=max_loc)][0])],
                                 self.X_train.index[self.X_train[self.forecast_dates].isin(tms.loc[(tms.index>=min_loc)*(tms.index<=max_loc)][0])]])
                    
        self.cv_indexes=index_output
        
    def get_hyper_parameters(self,array=0,overwrite=False,save=True):
        if self.verbose:
            print('Running parameter array '+str(array)+'.')
        
        
        if r'param_'+self.param_file_name+str(array)+'.csv' in os.listdir(self.parameter_directory) and not overwrite:
            self.best_params=pd.read_csv(self.parameter_directory+r'param_'+self.param_file_name+str(array)+'.csv')
            self.best_params=self.best_params.set_index(['Unnamed: 0'])
        elif r'param_'+self.param_file_name+'.csv' in os.listdir(self.parameter_directory) and not overwrite:
            self.best_params=pd.read_csv(self.parameter_directory+r'param_'+self.param_file_name+'.csv')
            self.best_params=self.best_params.set_index(['Unnamed: 0'])
        elif overwrite:
            self.forecast_date_param=self.param_arrays[array]
            self.param_use=self.forecast_date_param
            self.data_clean(self.forecast_date_param)
            start=time()
            exec("self."+self.algorithm+'()')
            if self.param_iter:
                self.bayes_search=BayesSearchCV(self.model, self.param_distributions,n_iter=self.param_iter,n_jobs=max(cpu_count()-1,1),
                                       n_points=max(cpu_count()-1,1),
                                       cv=self.cv_indexes,#iid=False,
                                       random_state=0,
                                       scoring={'mae':'neg_mean_absolute_error',
                                                'mse':'neg_mean_squared_error'}[self.objective],
                                       )
                #Generate the Params
                self.bayes_search.fit(self.X_train, self.Y_train[self.target],**self.fit_kwargs)
                self.best_params=pd.DataFrame(self.bayes_search.best_params_,index=[self.forecast_date_param])
                
                
                if save:
                    self.best_params.to_csv(self.parameter_directory+r'param_'+self.param_file_name+str(array)+'.csv')
                self.hyper_runtime=time()-start
        
    def get_predictions(self,array=0,save=True,feature_importance=None,
                        save_model=False, model_directory=''):
        self.feature_importance=feature_importance
        if self.verbose:
            print('Running prediction array '+str(array)+'.')
        self.forecast_date=self.pred_arrays[array]
        
        self.data_clean(self.forecast_date)
        
        if not self.algorithm in ['OLS','esLGBM','NHN']:
            self.param_use=np.max([x for x in self.param_arrays if x<=self.forecast_date])
            self.get_hyper_parameters(array=self.param_arrays.index(self.param_use),overwrite=False)
            if not self.param_use in self.best_params.index:
                raise ValueError('The necessary parameters for this array are not in the parameter file. Try re-running get_hyper_parameters with overwrite=True.')
        
        start=time()
        exec("self."+self.algorithm+'()')
        #fit the model
        # self.model.fit(np.array(self.X_train), np.array(self.Y_train[self.target]),**self.fit_kwargs)
        self.model.fit(self.X_train, self.Y_train[[self.target]],**self.fit_kwargs)
        # self.Y_test['prediction']=self.model.predict(np.array(self.X_test))
        self.Y_test['prediction']=self.model.predict(self.X_test)
        if save:
            self.Y_test.to_csv(self.prediction_directory+r'pred_'+self.pred_file_name+'_'+str(array)+'.csv',index=False)
        self.prediction_runtime=time()-start
        
        if feature_importance:
            self.get_feat_imp(array=array,save=save)
            
        if save_model:
            self.model.save_model(model_directory+self.pred_file_name+str(array)+'_')
        
    
    class Custom_Linear(BaseEstimator, ClassifierMixin):  
    
        def __init__(self, alpha=0, l1_ratio=0, huber_penalty=100):
            self.alpha=alpha
            self.l1_ratio=l1_ratio
            self.huber_penalty=huber_penalty
            self.classes_=' '
            self._estimator_type = "regressor"
            
        # def set_params(self,param_dict):
        #     for i in self.best_params:
        #         exec("self."+i+"="+param_dict[i][0])
            
            
        def huber_gradient(self,theta,x,y):
            #Calculate Current Errors
            errs=np.subtract(y,np.matmul(x,theta))
            
            #calculate the gradient. If huber_penalty is 0, it uses mae gradient
            if self.huber_penalty==0:
                huber_vec=(1/x.shape[0])*np.where(errs<0,1,np.where(errs>0,-1,0))
            else:
                #Get the cutoff for the huber loss
                huber_cutoff=np.percentile(np.abs(errs),self.huber_penalty)
                
                #Create a vector of values equal to the huber cutoff point
                huber_vec=np.ones((x.shape[0]))*huber_cutoff
                
                #Outside the huber cutoff, the gradient is equal to the (negative)
                #of the huber cutoff. Inside its equal to the negative of the errors
                huber_vec=(1/x.shape[0])*np.where(errs<-huber_cutoff,-huber_vec,
                       np.where(errs>huber_cutoff,huber_vec,
                              -errs))
                
            #Now add the elasticnet L1 and L2 gradients
            en_gradient=np.where(theta>0,self.alpha*(self.l1_ratio)+self.alpha*(1-self.l1_ratio)*np.abs(theta),
                             np.where(theta<0,-(self.alpha*(self.l1_ratio)+self.alpha*(1-self.l1_ratio)*np.abs(theta)),
                                      0))
            
            return np.matmul(x.T,huber_vec)+en_gradient
        
        def loss_en(self,theta,x,y):
            #Calculate the errors
            errs=np.subtract(y,np.matmul(x,theta))
            
            #calculate the loss. If huber_penalty is 0, it uses mae loss
            if self.huber_penalty==0:
                huber_loss=np.mean(np.abs(errs))
            else:
                #Get the cutoff for the huber loss
                huber_cutoff=np.percentile(np.abs(errs),self.huber_penalty)
                
                #Calculate huber loss for each observation then average
                huber_loss=np.mean(np.where(np.abs(errs)>huber_cutoff,
                                                      2*huber_cutoff*np.abs(errs)-huber_cutoff**2,
                              errs**2))
            
            #Add elastic net L1 and L2 penalties
            en_loss=self.alpha*(1-self.l1_ratio)*np.sum(np.abs(theta))+0.5*self.alpha*self.l1_ratio*np.sum(theta**2)
            return huber_loss+en_loss
        
        #This function is the soft thresholding operator and is a piece of the proximal algorithm
        def S(self,x,mu):
            out=[]
            #Loop over coefficients
            for ix in x:
                #If the gradient wants the coefficient to move past zero, then we
                #set it to zero. Otherwise shift the coefficient
                if ix>0 and mu<np.abs(ix):
                    out.append(ix-mu)
                elif ix<0 and mu<np.abs(ix):
                    out.append(ix+mu)
                elif mu>=np.abs(ix):
                    out.append(0)
            return np.array(out)
        
        #Proximal algorithm. This sets coefficients to zero and improves efficiency
        def prox_op_en(self,theta):
           return (1/(1+self.alpha*self.gamma*(1-self.l1_ratio)))*self.S(theta,self.l1_ratio*self.alpha*self.gamma)
        
        def fit(self,x,y,max_iter=1000):
            
            self.gamma=1
            
            #Intialize variables
            best_theta=np.zeros(x.shape[1])
            theta_0=best_theta
            converge=np.sum(np.abs(self.huber_gradient(theta_0,x,y.values.ravel())))
            converge_past=0
            best_loss=converge
            m=0
            
            #Run gradient descent Define convergence one gradient stops getting closer
            #to zero. Stop if hit max iterations, and require at least 5 iterations
            while (np.abs(converge-converge_past)>1e-10 and m<max_iter and self.gamma>1e-9) or m<5:
                #Store old convergence value
                converge_past=converge
                
                #calculate gradient and shift parameter values
                theta_bar=theta_0-self.gamma*self.huber_gradient(theta_0,x,y.values.ravel())
                
                #Run parameters through proximal algorithm to set coefficients to zero
                theta_0=self.prox_op_en(theta_bar)
                
                #Calculate new convergence value (sum of abs(gradient))
                converge=np.sum(np.abs(self.huber_gradient(theta_0,x,y.values.ravel())))
                
                #If our gradient improved compared to the last loop, store the parameters
                #and best loss value
                if converge<best_loss :
                    best_theta=theta_0
                    best_loss=converge
                
                
                #Logic based learning rate acceleration:
                #If our gradient is worse than the previous loop, our learning rate (gamma)
                #is too high. So, decrease it and try again
                if (converge>best_loss):
                    #Decrease learning rate
                    self.gamma=self.gamma*0.9
                    
                    #revert to previous theta, and reset convergence values to previous
                    #state
                    theta_0=best_theta
                    converge=np.sum(np.abs(self.huber_gradient(theta_0,x,y.values.ravel())))
                    converge_past=0
                    
                #If our gradient improved, but only by a little bit, then we can 
                #Accelerate our convergence by increasing the learning rate
                elif 1-np.abs(converge/converge_past)>1e-3:
                    self.gamma=self.gamma*1.01
                
                #This prints if we are about to hit max iterations
                if m>max_iter-20:
                    print('This array is going to hit the max iterations')
                    print('Converge '+str(m)+': '+str(converge))
                m=m+1
                
            self.coef_=theta_0
            return self
        
       
        def predict(self, X):
            return np.matmul(X,self.coef_)
        
        def score(self, X, y=None):
            #Calculate the errors
            errs=np.subtract(y.values.ravel(),np.matmul(X,self.coef_))
            
            #calculate the loss. If huber_penalty is 0, it uses mae loss
            if self.huber_penalty==0:
                huber_loss=np.mean(np.abs(errs))
            else:
                #Get the cutoff for the huber loss
                huber_cutoff=np.percentile(np.abs(errs),self.huber_penalty)
                
                #Calculate huber loss for each observation then average
                huber_loss=np.mean(np.where(np.abs(errs)>huber_cutoff,
                                                      2*huber_cutoff*np.abs(errs)-huber_cutoff**2,
                              errs**2))
            
            return -huber_loss
    
    def OLS(self):
        self.param_iter=None
        
        if self.objective=='mse':
            self.model =  LinearRegression(fit_intercept=False)
            
        if self.objective=='mae':
            self.model=self.Custom_Linear(alpha=0,
                                   l1_ratio=0,huber_penalty=0)
            
        if self.objective=='huber':
            self.model=self.Custom_Linear(alpha=0,
                                   l1_ratio=0,huber_penalty=99)
        
    def EN(self):
        self.param_distributions = {
        "alpha": Real(1e-7, 1e4,prior='log-uniform'),
        "l1_ratio": Real(0, 1,prior='uniform')}#, 
        self.param_iter=100
        use_params={}
        if self.param_use in self.best_params.index:
            use_params=dict(self.best_params.loc[self.param_use,:])
        if self.objective=='mse':
            self.model =  ElasticNet(**use_params,
                               fit_intercept=True,
                               max_iter=10000)
            
        if self.objective=='mae':
            self.model=self.Custom_Linear(**use_params,huber_penalty=0)
            
        if self.objective=='huber':
            self.model=self.Custom_Linear(**use_params,huber_penalty=99)
            
    def Ridge(self):
        self.param_distributions = {
        "alpha": Real(1e-7, 1e4,prior='log-uniform')}#, 
        
        self.param_iter=100
        use_params={}
        if self.param_use in self.best_params.index:
            use_params=dict(self.best_params.loc[self.param_use,:])
        if self.objective=='mse':
            self.model =  ElasticNet(**use_params,
                               fit_intercept=True,
                               max_iter=10000,
                                   l1_ratio=0)
            
        if self.objective=='mae':
            self.model=self.Custom_Linear(**use_params,huber_penalty=0,
                                   l1_ratio=0)
            
        if self.objective=='huber':
            self.model=self.Custom_Linear(**use_params,huber_penalty=99,
                                   l1_ratio=0)
            
            
    def Lasso(self):
        self.param_distributions = {
        "alpha": Real(1e-7, 1e4,prior='log-uniform')}#, 
        
        self.param_iter=100
        use_params={}
        if self.param_use in self.best_params.index:
            use_params=dict(self.best_params.loc[self.param_use,:])
        if self.objective=='mse':
            self.model =  ElasticNet(**use_params,
                               fit_intercept=True,
                               max_iter=10000,
                                   l1_ratio=1)
            
        if self.objective=='mae':
            self.model=self.Custom_Linear(**use_params,huber_penalty=0,
                                   l1_ratio=1)
            
        if self.objective=='huber':
            self.model=self.Custom_Linear(**use_params,huber_penalty=99,
                                   l1_ratio=1)
            
    def LGBM(self):
        self.param_distributions = {
        "learning_rate": Real(1e-4, 1,prior='log-uniform'),
        'max_depth':Integer(1,3)}#, 
        self.param_iter=50
        
        use_params={}
        if self.param_use in self.best_params.index:
            use_params=dict(self.best_params.loc[self.param_use,:])
            use_params['max_depth']=int(use_params['max_depth'])
        
        if self.objective=='huber':
            self.model = lgb.LGBMRegressor(**use_params,n_estimators=2000, boosting_type='gbdt',data_sample_strategy='goss',
                                           n_jobs=-1, 
                                     metrics=['get_grad_hess'],
                        min_child_samples=5,objective=self.get_grad_hess,random_state=0)
            self.fit_kwargs={'eval_metric':self.huber_loss}
        else:
            self.model = lgb.LGBMRegressor(**use_params,n_estimators=2000, boosting_type='gbdt',data_sample_strategy='goss',
                                           n_jobs=-1, 
                        min_child_samples=5,objective=self.objective,random_state=0)
            
    def RF(self):
        self.param_distributions = {
            'colsample_bynode':  Real(.001, .999,prior='log-uniform'),
            'max_depth': Integer(1,8),#8
            # 'subsample': Real(.001, .999,prior='log-uniform'),
        }
        self.param_iter=50
        
        use_params={}
        if self.param_use in self.best_params.index:
            use_params=dict(self.best_params.loc[self.param_use,:])
            use_params['max_depth']=int(use_params['max_depth'])
        
        
        if self.objective=='huber':
            self.model = lgb.LGBMRegressor(**use_params,n_estimators=2000, boosting_type='rf',n_jobs=-1, 
                                     metrics=['get_grad_hess'],
                        min_child_samples=5,objective=self.get_grad_hess,random_state=0,
                        subsample_freq=1,#subsample=0.5
                        )
            self.fit_kwargs={'eval_metric':self.huber_loss}
        else:
            self.model = lgb.LGBMRegressor(**use_params,n_estimators=2000, boosting_type='rf',n_jobs=-1, 
                        min_child_samples=5,objective=self.objective,random_state=0,
                        subsample_freq=1,#subsample=0.5
                        )
            self.fit_kwargs={'eval_metric':self.objective}
            
    def esLGBM(self):
        self.param_iter=None
        from Custom_esLGBM import Custom_esLGBM
        
        self.model=Custom_esLGBM(learning_rate=0.0025,#parameters.loc[0,'learning_rate'],
                               objective=self.objective,
                               cv_ind=self.cv_indexes)
    
    def NN(self):
        self.param_distributions={'dropout_u':Real(.2, .4,prior='uniform'),
                            'l1l2penal':Real(.001, .1,prior='log-uniform'),
                            # 'l1l2penal':Real(.5, 1,prior='uniform'),
                         # 'n_layers':Integer(1,3,prior='uniform')
                         }
        from NN_Custom_sklearn import NN
        self.param_iter=50
        
        use_params={}
        if self.param_use in self.best_params.index:
            use_params=dict(self.best_params.loc[self.param_use,:])
        
        if not 'Keras' in os.listdir():
            os.mkdir('Keras')
            
        
        self.fit_kwargs['dumploc']=r'C:\Users\Harri\Documents\Keras' if socket.gethostname() in ['Desktop','Laptop-Ham'] else 'Keras'
        self.fit_kwargs['gap_']=0 if not self.cv_gap else int(self.gap)
        
        if 'multi_target_var' in list(self.fit_kwargs.keys()):
            self.model = NN(objective=self.objective,
                            learning_rate=0.02,
                             # multi_target_var=self.fit_kwargs['multi_target_var'],
                            train_dta=self.Y_train[[self.forecast_dates,self.fit_kwargs['multi_target_var']]],
                            test_dta=self.Y_test[[self.forecast_dates,self.fit_kwargs['multi_target_var']]],
                            time_var=self.forecast_dates,
                            **use_params)
        else:
            self.model = NN(objective=self.objective,
                        learning_rate=0.02,
                        **use_params)
    def NHN(self):
        
        from NN_Custom_sklearn import NN
        
        if not 'Keras' in os.listdir():
            os.mkdir('Keras')
        
        NN_Params = {'Dropout': [0.2, 0.4], 'l1l2': [0.5, 1]}
        
        def NNGridSearchWrapper(X, Y,X_test, archi=None,  params=None, refit=None, dumploc=None,multi_target_var=None,
                                gap_=None,
                                Y_train=None,
                                Y_test=None,
                                forecast_dates=None):
            # Construct grid of parameters from dictionary, containing param ranges
            paramgrid = list(ParameterGrid(params))

            # Loop over all param grid combinations and save val_loss
            val_loss = list()
            for i, param_i in enumerate(paramgrid):
                mod= NN(objective=self.objective,
                        learning_rate=0.02,
                        time_var=forecast_dates,
                        dropout_u=param_i['Dropout'],
                        l1l2penal=param_i['l1l2'],
                        ensemble=1,
                        train_dta=Y_train[[forecast_dates,multi_target_var]],
                        test_dta=Y_test[[forecast_dates,multi_target_var]],
                        )
                mod.fit(X,Y,dumploc=dumploc,
                gap_=int(gap_),
                archi=archi,)
                
                val_loss.append(mod.losses[0])
            
            # Determine best model according to grid-search val_loss
            bestm = np.argmin(val_loss)
            
            # Fit best model again
            mod= NN(
                        objective=self.objective,
                        learning_rate=0.02,
                        time_var=forecast_dates,
                        ensemble=1,
                        dropout_u=paramgrid[bestm]['Dropout'],
                        l1l2penal=paramgrid[bestm]['l1l2'],
                        train_dta=Y_train[[forecast_dates,multi_target_var]],
                        test_dta=Y_test[[forecast_dates,multi_target_var]],
                        )
                        
            
            mod.fit(X,Y,dumploc=dumploc,
                    gap_=int(gap_),
                    archi=archi,)
            
            return mod.predict(X_test), val_loss[bestm]
        
        class NHN():
            def __init__(self,n_layers=1):
                """
                Called when initializing the classifier
                """
                self.classes_=' '
                self._estimator_type = "regressor"
                self.n_layers=n_layers
                
            def fit(self,X,Y,X_test=None,
                    ensemble=2,archi=None,multi_target_var=None,dumploc=None,gap_=None,
                    Y_train=None,Y_test=None,forecast_dates=None):
                self.output=[NNGridSearchWrapper(X=X,Y=Y,X_test=X_test,params=NN_Params,archi=archi,multi_target_var=multi_target_var,
                                            dumploc=dumploc,gap_=gap_,
                                            Y_train=Y_train,
                                            Y_test=Y_test,
                                            forecast_dates=forecast_dates,
                                            ) for i in np.arange(ensemble)]
                losses=[self.output[i][1] for i in np.arange(ensemble)]
                preds=[self.output[i][0] for i in np.argpartition(losses,min(ensemble-1,10))[:min(ensemble,10)]]
                
                self.oos=np.array(preds).mean(axis=0)
            def predict(self,X_test):
                return self.oos
        self.fit_kwargs['gap_']=int(0) if not self.cv_gap else int(self.gap)
        self.fit_kwargs['dumploc']=r'C:\Users\Harri\Documents\Keras' if socket.gethostname() in ['Desktop','Laptop-Ham'] else 'Keras'
        self.fit_kwargs['Y_train']=self.Y_train
        self.fit_kwargs['Y_test']=self.Y_test
        self.fit_kwargs['forecast_dates']=self.forecast_dates
        self.fit_kwargs['X_test']=self.X_test
        self.model=NHN()
        
    def get_grad_hess(y_true, y_pred):
        ###### Hessian
        huber_penalty=99
        residual = y_true-y_pred
        huber_cutoff=np.percentile(np.abs(residual),huber_penalty)
        huber_grad=(1/len(residual))*np.where(residual<=-huber_cutoff,-huber_cutoff,
                      np.where(residual>=huber_cutoff,huber_cutoff,
                              -residual))
        
        huber_hess=(1/len(residual))*np.where(residual<=-huber_cutoff,0,
                      np.where(residual>=huber_cutoff,0,
                              1))
        
        return huber_grad, huber_hess
    
    
    def huber_loss(y_true, y_pred):
        huber_penalty=99
        residual = y_true - y_pred
        huber_cutoff=np.percentile(np.abs(residual),huber_penalty)
        huber_loss=(1/len(residual))*np.sum(np.where(np.abs(residual)>huber_cutoff,
                                                  2*huber_cutoff*np.abs(residual)-huber_cutoff**2,
                          residual**2))
        return 'huber_loss', huber_loss,True
    
    def combine_parameters(self):
        if not self.algorithm in ['OLS','esLGBM','NHN']:
            needed=[r'param_'+self.param_file_name+str(x)+'.csv' for x in np.arange(self.n_param_arrays)]
            have=os.listdir(self.parameter_directory)
            if len(list(set(needed)&set(have)))==self.n_param_arrays:
                self.best_params=pd.concat([pd.read_csv(self.parameter_directory+x) for x in needed]).set_index(['Unnamed: 0'])
                self.best_params.to_csv(self.parameter_directory+r'param_'+self.param_file_name+'.csv')
                for x in needed:
                    try:
                        os.remove(self.parameter_directory+x)
                    except PermissionError:
                        sleep(3)
                        os.remove(self.parameter_directory+x)
                        
            elif self.algorithm!='OLS':
                self.missing_params=pd.DataFrame([[self.parameter_directory+x for x in list(set(needed)-set(have))]])
                self.missing_param_arrays=[x.replace('param_'+self.param_file_name,'').replace('.csv','') for x in list(set(needed)-set(have))]
                if not r'param_'+self.param_file_name+'.csv' in os.listdir(self.parameter_directory):
                    print('Missing files for parameter combine. See missing_params.')
        
    def combine_predictions(self):
        
        needed=[r'pred_'+self.pred_file_name+'_'+str(x)+'.csv' for x in np.arange(self.n_pred_arrays)]
        have=os.listdir(self.prediction_directory)
        if len(list(set(needed)&set(have)))==self.n_pred_arrays:
            self.pred_out=pd.concat([pd.read_csv(self.prediction_directory+x) for x in needed])
            self.pred_out.to_csv(self.result_directory+r'preds_'+self.result_file_name+'.csv',index=False)
            for x in needed:
                os.remove(self.prediction_directory+x)
            if self.feature_importance:
                self.coef_out=pd.concat([pd.read_csv(self.prediction_directory+x.replace('pred_','coef_')) for x in needed])
                self.coef_out.to_csv(self.result_directory+r'coefs_'+self.result_file_name+'.csv',index=False)
                for x in needed:
                    os.remove(self.prediction_directory+x.replace('pred_','coef_'))
        else:
            self.missing_preds=pd.DataFrame([[self.prediction_directory+x for x in list(set(needed)-set(have))]])
            self.missing_pred_arrays=[x.replace('pred_'+self.pred_file_name+'_','').replace('.csv','') for x in list(set(needed)-set(have))]
            print('Missing files for prediction combine. See missing_preds.')
    
    def pca_predictors(self):
        merge_vars=list(set(self.identifiers)&set(self.df_pca.columns))
        #Pull out test set
        train_pca=pd.merge(self.X_train[merge_vars],self.df_pca,how='left',on=merge_vars).set_index(merge_vars)
        
        #X_train Set
        test_pca=pd.merge(self.X_test[merge_vars],self.df_pca,how='left',on=merge_vars).set_index(merge_vars)
        
        pca = PCA()
        df_pc=pd.DataFrame(pca.fit_transform(train_pca)[:,:self.n_pcs],index=train_pca.index,columns=['pc'+str(x) for x in np.arange(self.n_pcs)]).reset_index()
        df_pc_test=pd.DataFrame(pca.transform(test_pca)[:,:self.n_pcs],index=test_pca.index,columns=['pc'+str(x) for x in np.arange(self.n_pcs)]).reset_index()
        
        if df_pc[merge_vars].drop_duplicates().shape[0]!=df_pc.shape[0]:
            print('Principal component dataframe does not have unique identifiers which may crash the code. If it keeps going dont worry about it though...')
        self.X_train=pd.merge(self.X_train, df_pc, how='left',on=merge_vars)
        self.X_test=pd.merge(self.X_test, df_pc_test, how='left',on=merge_vars)
        
        if self.predictors:
            self.predictors.extend(['pc'+str(x) for x in np.arange(self.n_pcs)])
    
    def get_feat_imp(self,array,save=True):
        def mae(pred,actual):
            return np.mean(np.abs(np.subtract(pred,actual)))
        mae_full=mae(self.model.predict(self.X_train),self.Y_train[[self.target]].values.ravel())
         
        self.coefs=[]
        for icol in self.X_train.columns:
            temp=self.X_train.copy()
            temp[icol]=0
            self.coefs.append(mae(self.model.predict(temp),self.Y_train[[self.target]].values.ravel())-mae_full)
        
        self.coefs=pd.DataFrame(self.coefs,index=self.X_test.columns, columns=['Coefficient'])
        self.coefs[self.forecast_dates]=self.forecast_date
        if save:
            self.coefs.to_csv(self.prediction_directory+r'coef_'+self.pred_file_name+'_'+str(array)+'.csv')