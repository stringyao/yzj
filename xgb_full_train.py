VALIDATE = True

MAX_ROUNDS = 1000
EARLY_STOP = 50
OPT_ROUNDS = 26

FULL_OUTFILE = 'xgb_parallel_sub_full.csv'
VALID_OUTFILE = 'xgb_parallel_validation.csv'

import pandas as pd
import time
import numpy as np
import xgboost as xgb
import gc
 
path = '/usr/talkingdata/'

print('load training data...')
train_df = pd.read_pickle(path + 'training.pkl.gz', 'gzip')
train_df.click_time = pd.to_datetime(train_df.click_time)
gc.collect()

print('load validating data...')
validation_df = pd.read_pickle(path + 'validation.pkl.gz', 'gzip')
validation_df.click_time = pd.to_datetime(validation_df.click_time)
gc.collect()

data_full = pd.concat([train_df, validation_df])

del train_df
del validation_df
gc.collect()

print('data prep...')
most_freq_hours_in_test_data = [4, 5, 9, 10, 13, 14]
least_freq_hours_in_test_data = [6, 11, 15]

def prep_data( df ):
    
    # Adding some new features
    
    df['hour'] = pd.to_datetime(df.click_time).dt.hour.astype('uint8')
    df['day'] = pd.to_datetime(df.click_time).dt.day.astype('uint8')
    #df['minute'] = pd.to_datetime(df.click_time).dt.minute.astype('uint8') # add minute
    #df['second'] = pd.to_datetime(df.click_time).dt.second.astype('uint8') # add second 
    
    # print('group by : ip_nextClick')
    # df['ip_nextClick'] = df[['ip', 'click_time']].groupby(['ip']).click_time.transform(lambda x: x.diff().shift(-1)).dt.seconds
    # df['ip_nextClick'] = df['ip_nextClick'].astype('float16')
    # gc.collect()
    # print( df.info() ) 
    
    # print('group by : ip_app_nextClick')
    # df['ip_app_nextClick'] = df[['ip', 'app', 'click_time']].groupby(['ip', 'app']).click_time.transform(lambda x: x.diff().shift(-1)).dt.seconds
    # df['ip_app_nextClick'] = df['ip_app_nextClick'].astype('float16')
    # gc.collect()
    # print( df.info() ) 
    
    # print('group by : ip_channel_nextClick')
    # df['ip_channel_nextClick'] = df[['ip', 'channel', 'click_time']].groupby(['ip', 'channel']).click_time.transform(lambda x: x.diff().shift(-1)).dt.seconds
    # df['ip_channel_nextClick'] = df['ip_channel_nextClick'].astype('float16')
    # gc.collect()
    # print( df.info() ) 

    # print('group by : ip_os_nextClick')
    # df['ip_os_nextClick'] = df[['ip', 'os', 'click_time']].groupby(['ip', 'os']).click_time.transform(lambda x: x.diff().shift(-1)).dt.seconds
    # df['ip_os_nextClick'] = df['ip_os_nextClick'].astype('float16')
    # gc.collect()
    # print( df.info() ) 
    
    df.drop(['click_time'], axis=1, inplace=True)
    gc.collect()
    

    
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
    # print('group by: ip_app_channel')
    # gp = df[['ip','app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})
    # df = df.merge(gp, on=['ip','app'], how='left')
    # del gp
    # df['ip_app_count'] = df['ip_app_count'].astype('uint16')
    # gc.collect()
    # print( df.info() )

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


print('preprocessing training data...')
train_df = prep_data( data_full )
del data_full
gc.collect()



print("training preparation ...")

metrics = 'auc'
params = {'eta': 0.3,
          'tree_method': "hist",
          'grow_policy': "lossguide",
          'max_leaves': 1400,  
          'max_depth': 0, 
          'subsample': 0.9, 
          'colsample_bytree': 0.7, 
          'colsample_bylevel':0.7,
          'min_child_weight':0,
          'alpha':4,
          'objective': 'binary:logistic', 
          'scale_pos_weight':9,
          'eval_metric': 'auc', 
          'nthread':8,
          'random_state': 99, 
          'silent': True
          }

target = 'is_attributed'
predictors = ['app','device','os', 'channel', 'hour', 'nip_day_test_hh', 'nip_day_hh',
              'nip_hh_os', 'nip_hh_app', 'nip_hh_dev']  #, 'ip_freq', 'app_freq', 'device_freq', 'os_freq', 'channel_freq'] # add minute and second 
#predictors = predictors + ['minute', 'second'] + ['ip_app_count'] + ['ip_nextClick', 'ip_app_nextClick', 'ip_channel_nextClick', 'ip_os_nextClick']# + ['ip_app_os_count']
categorical = ['app', 'device', 'os', 'channel', 'hour']



print("Training...")

print("train size: ", len(train_df))

gc.collect()



num_boost_round=OPT_ROUNDS

dtrain = xgb.DMatrix(train_df[predictors].values, train_df[target].values)

del train_df
gc.collect()

watchlist = [(dtrain, 'train')]
model = xgb.train(params, dtrain, num_boost_round, watchlist, maximize=True, verbose_eval=1)

outfile = FULL_OUTFILE

del dtrain

gc.collect()

print('load test...')

outfile = FULL_OUTFILE

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }
        
test_cols = ['ip','app','device','os', 'channel', 'click_time', 'click_id'] 

test_df = pd.read_csv(path + "test.csv", dtype=dtypes, usecols=test_cols)

test_df = prep_data( test_df )
gc.collect()

sub = pd.DataFrame()
sub['click_id'] = test_df['click_id']

print("Predicting...")
sub['is_attributed'] = model.predict(test_df[predictors])
print("writing...")
sub.to_csv(outfile, index=False, float_format='%.9f')
print("done...")
print(sub.info())
