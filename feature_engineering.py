# add new features: Grouping by ['ip', 'app', 'channel'], and aggregating hour with mean
print('group by: ip_app_channel')
gp = df[['ip','app', 'channel', 'hour']].groupby(by=['ip', 'app', 'channel'])[['hour']].mean().reset_index().rename(index=str, columns={'hour': 'ip_app_channel_mean_hour'}).astype('float32')
df = df.merge(gp, on=['ip', 'app', 'channel'], how='left')
del gp
gc.collect()
print( df.info() )
