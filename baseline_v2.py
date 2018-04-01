import pandas as pd 
import gc

path = '../input/'

data_cols = ['ip','app','device','os', 'channel', 'click_time', 'is_attributed'] 

print('load training data...')
train_df = pd.read_pickle('../input/training.pkl.gz', 'gzip')
train_df.click_time = pd.to_datetime(train_df.click_time)
gc.collect()

print('load validating data...')
validation_df = pd.read_pickle('../input/validation.pkl.gz', 'gzip')
validation_df.click_time = pd.to_datetime(validation_df.click_time)
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
    
    print('group by : ip_nextClick')
    df['ip_nextClick'] = df[['ip', 'click_time']].groupby(['ip']).click_time.transform(lambda x: x.diff().shift(-1)).dt.seconds
    df['ip_nextClick'] = df['ip_nextClick'].astype('float16')
    gc.collect()
    print( df.info() ) 
    
    print('group by : ip_app_nextClick')
    df['ip_app_nextClick'] = df[['ip', 'app', 'click_time']].groupby(['ip', 'app']).click_time.transform(lambda x: x.diff().shift(-1)).dt.seconds
    df['ip_app_nextClick'] = df['ip_app_nextClick'].astype('float16')
    gc.collect()
    print( df.info() ) 
    
    print('group by : ip_channel_nextClick')
    df['ip_channel_nextClick'] = df[['ip', 'channel', 'click_time']].groupby(['ip', 'channel']).click_time.transform(lambda x: x.diff().shift(-1)).dt.seconds
    df['ip_channel_nextClick'] = df['ip_channel_nextClick'].astype('float16')
    gc.collect()
    print( df.info() ) 

    print('group by : ip_os_nextClick')
    df['ip_os_nextClick'] = df[['ip', 'os', 'click_time']].groupby(['ip', 'os']).click_time.transform(lambda x: x.diff().shift(-1)).dt.seconds
    df['ip_os_nextClick'] = df['ip_os_nextClick'].astype('float16')
    gc.collect()
    print( df.info() ) 
    
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


print('preprocessing training data...')
train_df = prep_data( train_df )
gc.collect()

print('preprocessing validating data...')
validation_df = prep_data( validation_df )
gc.collect()

print("vars and data type: ")
train_df.info()
validation_df.info()

print("training preparation ...")
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
predictors = predictors + ['minute', 'second'] + ['ip_app_count'] + ['ip_nextClick', 'ip_app_nextClick', 'ip_channel_nextClick', 'ip_os_nextClick']# + ['ip_app_os_count']
categorical = ['app', 'device', 'os', 'channel', 'hour']

if VALIDATE:

    print(train_df.info())
    print(validation_df.info())

    print("train size: ", len(train_df))
    print("valid size: ", len(validation_df))

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

    xgvalid = lgb.Dataset(validation_df[predictors].values, label=validation_df[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical
                          )
    del validation_df
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
#test_cols = test_cols + ['minute', 'second'] + ['ip_app_count']

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

