

VALIDATE = False

MAX_ROUNDS = 1000
EARLY_STOP = 50
OPT_ROUNDS = 906

FULL_OUTFILE = 'sub_lgbm_r_to_python_nocv.csv'
VALID_OUTFILE = 'sub_lgbm_r_to_python_withcv.csv'

import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split 
import lightgbm as lgb

path = '/usr/talkingdata/'

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }

print('load train...')
train_cols = ['ip','app','device','os', 'channel', 'click_time', 'is_attributed']
train_df = pd.read_csv(path+"train.csv", skiprows=range(1,84903891), nrows=100000000,dtype=dtypes, usecols=train_cols)

import gc

len_train = len(train_df)

gc.collect()

print('data prep...')

most_freq_hours_in_test_data = [4, 5, 9, 10, 13, 14]
least_freq_hours_in_test_data = [6, 11, 15]


def prep_data( df ):
    
    # Adding some new features
    
    df['hour'] = pd.to_datetime(df.click_time).dt.hour.astype('uint8')
    df['day'] = pd.to_datetime(df.click_time).dt.day.astype('uint8')
    df['minute'] = pd.to_datetime(df.click_time).dt.minute.astype('uint8') # add minute
    df['second'] = pd.to_datetime(df.click_time).dt.second.astype('uint8') # add second 
    df.drop(['click_time'], axis=1, inplace=True)
    gc.collect()
    
    ##############################################################################################################################################
    ##############################################################################################################################################
    # Add frequency, which might lead to some overfitting, or at least might take a long time 
    # Maybe it's a better idea to save the processed data to hdf5
    
    # FREQUENCY_COLUMNS = ['ip', 'app', 'device', 'os', 'channel']
    
    # # Find frequency of is_attributed for each unique value in column
    # freqs = {}
    # for col in FREQUENCY_COLUMNS:
    #     print(f">> Calculating frequency for: {col}")
    
    #     # Get counts, sums and frequency of is_attributed
    #     freq_df = pd.DataFrame({
    #         'sums': df.groupby(col)['is_attributed'].sum(),
    #         'counts': df.groupby(col)['is_attributed'].count()
    #     })
    #     freq_df.loc[:, 'freq'] = freq_df.sums / freq_df.counts
        
    #     # If we have less than 3 observations, e.g. for an IP, then assume freq of 0
    #     freq_df.loc[freq_df.counts <= 3, 'freq'] = 0        
        
    #     # Add to X_total
    #     df[col+'_freq'] = df[col].map(freq_df['freq'])
        
    # del freq_df
    # gc.collect()
        
    ##############################################################################################################################################
    ##############################################################################################################################################
    
    df['in_test_hh'] = (   3 
                         - 2*df['hour'].isin(  most_freq_hours_in_test_data ) 
                         - 1*df['hour'].isin( least_freq_hours_in_test_data ) ).astype('uint8')
    print( df.info() )

    print('group by : ip_day_test_hh')
    gp = df[['ip', 'day', 'in_test_hh', 'channel']].groupby(by=['ip', 'day',
             'in_test_hh'])[['channel']].count().reset_index().rename(index=str, 
             columns={'channel': 'nip_day_test_hh'})
    df = df.merge(gp, on=['ip','day','in_test_hh'], how='left')
    del gp
    df.drop(['in_test_hh'], axis=1, inplace=True)
    print( "nip_day_test_hh max value = ", df.nip_day_test_hh.max() )
    df['nip_day_test_hh'] = df['nip_day_test_hh'].astype('uint32')
    gc.collect()
    print( df.info() )

    print('group by : ip_day_hh')
    gp = df[['ip', 'day', 'hour', 'channel']].groupby(by=['ip', 'day', 
             'hour'])[['channel']].count().reset_index().rename(index=str, 
             columns={'channel': 'nip_day_hh'})
    df = df.merge(gp, on=['ip','day','hour'], how='left')
    del gp
    print( "nip_day_hh max value = ", df.nip_day_hh.max() )
    df['nip_day_hh'] = df['nip_day_hh'].astype('uint16')
    gc.collect()
    print( df.info() )

    print('group by : ip_hh_os')
    gp = df[['ip', 'day', 'os', 'hour', 'channel']].groupby(by=['ip', 'os', 'day',
             'hour'])[['channel']].count().reset_index().rename(index=str, 
             columns={'channel': 'nip_hh_os'})
    df = df.merge(gp, on=['ip','os','hour','day'], how='left')
    del gp
    print( "nip_hh_os max value = ", df.nip_hh_os.max() )
    df['nip_hh_os'] = df['nip_hh_os'].astype('uint16')
    gc.collect()
    print( df.info() )

    print('group by : ip_hh_app')
    gp = df[['ip', 'app', 'hour', 'day', 'channel']].groupby(by=['ip', 'app', 'day',
             'hour'])[['channel']].count().reset_index().rename(index=str, 
             columns={'channel': 'nip_hh_app'})
    df = df.merge(gp, on=['ip','app','hour','day'], how='left')
    del gp
    print( "nip_hh_app max value = ", df.nip_hh_app.max() )
    df['nip_hh_app'] = df['nip_hh_app'].astype('uint16')
    gc.collect()
    print( df.info() )

    print('group by : ip_hh_dev')
    gp = df[['ip', 'device', 'hour', 'day', 'channel']].groupby(by=['ip', 'device', 'day',
             'hour'])[['channel']].count().reset_index().rename(index=str, 
             columns={'channel': 'nip_hh_dev'})
    df = df.merge(gp, on=['ip','device','day','hour'], how='left')
    del gp
    print( "nip_hh_dev max value = ", df.nip_hh_dev.max() )
    df['nip_hh_dev'] = df['nip_hh_dev'].astype('uint32')
    gc.collect()
    print( df.info() )
    
    # add new features: # of clicks for each ip-app combination
    print('group by: ip_app_channel')
    gp = df[['ip','app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})
    df = df.merge(gp, on=['ip','app'], how='left')
    del gp
    df['ip_app_count'] = df['ip_app_count'].astype('uint16')
    gc.collect()
    print( df.info() )

    # add new features: # of clicks for each ip-app-os combination
    # print('group by: ip_app_os')
    # gp = df[['ip','app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_os_count'})
    # df = df.merge(gp, on=['ip','app', 'os'], how='left')
    # del gp
    # df['ip_app_os_count'] = df['ip_app_os_count'].astype('uint16')
    # gc.collect()
    # print( df.info() )

    df.drop( ['ip','day'], axis=1, inplace=True )
    gc.collect()
    print( df.info() )
    
    return( df )

#---------------------------------------------------------------------------------

print( "Train info before: ")
print( train_df.info() )
train_df = prep_data( train_df )
gc.collect()
print( "Train info after: ")
print( train_df.info() )

print("vars and data type: ")
train_df.info()

metrics = 'auc'
lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric':metrics,
        'learning_rate': 0.1,
        'num_leaves': 7,  # we should let it be smaller than 2^(max_depth)
        'max_depth': 4,  # -1 means no limit
        'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 100,  # Number of bucketed bin for feature values
        'subsample': 0.7,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.7,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'nthread': 8,
        'verbose': 0,
        'scale_pos_weight':99.7, # because training data is extremely unbalanced 
        'metric':metrics
}

target = 'is_attributed'
predictors = ['app','device','os', 'channel', 'hour', 'nip_day_test_hh', 'nip_day_hh',
              'nip_hh_os', 'nip_hh_app', 'nip_hh_dev']  #, 'ip_freq', 'app_freq', 'device_freq', 'os_freq', 'channel_freq'] # add minute and second 
predictors = predictors + ['minute', 'second'] + ['ip_app_count']# + ['ip_app_os_count']
categorical = ['app', 'device', 'os', 'channel', 'hour']

print(train_df.head(5))

if VALIDATE:

    train_df, val_df = train_test_split( train_df, train_size=.95, shuffle=False )

    print(train_df.info())
    print(val_df.info())

    print("train size: ", len(train_df))
    print("valid size: ", len(val_df))

    gc.collect()

    print("Training...")

    num_boost_round=MAX_ROUNDS
    early_stopping_rounds=EARLY_STOP

    xgtrain = lgb.Dataset(train_df[predictors].values, label=train_df[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical
                          )
    del train_df
    gc.collect()

    xgvalid = lgb.Dataset(val_df[predictors].values, label=val_df[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical
                          )
    del val_df
    gc.collect()

    evals_results = {}

    bst = lgb.train(lgb_params, 
                     xgtrain, 
                     valid_sets= [xgvalid], 
                     valid_names=['valid'], 
                     evals_result=evals_results, 
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=10, 
                     feval=None)

    n_estimators = bst.best_iteration

    print("\nModel Report")
    print("n_estimators : ", n_estimators)
    print(metrics+":", evals_results['valid'][metrics][n_estimators-1])
    
    outfile = VALID_OUTFILE
    
    del xgvalid

else:

    print(train_df.info())

    print("train size: ", len(train_df))

    gc.collect()

    print("Training...")

    num_boost_round=OPT_ROUNDS

    xgtrain = lgb.Dataset(train_df[predictors].values, label=train_df[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical
                          )
    del train_df
    gc.collect()

    bst = lgb.train(lgb_params, 
                     xgtrain, 
                     num_boost_round=num_boost_round,
                     verbose_eval=10, 
                     feval=None)
                     
    outfile = FULL_OUTFILE

del xgtrain
gc.collect()

print('load test...')
test_cols = ['ip','app','device','os', 'channel', 'click_time', 'click_id'] 
test_df = pd.read_csv(path+"test.csv", dtype=dtypes, usecols=test_cols)

test_df = prep_data( test_df )
gc.collect()

sub = pd.DataFrame()
sub['click_id'] = test_df['click_id']

print("Predicting...")
sub['is_attributed'] = bst.predict(test_df[predictors])
print("writing...")
sub.to_csv(outfile, index=False, float_format='%.9f')
print("done...")
print(sub.info())
