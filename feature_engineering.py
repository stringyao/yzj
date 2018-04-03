# add new features: Grouping by ['ip', 'app', 'channel'], and aggregating hour with mean
print('group by: ip_app_channel')
gp = df[['ip','app', 'channel', 'hour']].groupby(by=['ip', 'app', 'channel'])[['hour']].mean().reset_index().rename(index=str, columns={'hour': 'ip_app_channel_mean_hour'}).astype('float32')
df = df.merge(gp, on=['ip', 'app', 'channel'], how='left')
del gp
gc.collect()
print( df.info() )


print('group by: ip_app_channel')
gp = df[['device','hour']].groupby(by=['device'])[['hour']].count().reset_index().rename(index=str, columns={'hour': 'nDiv_h'})
df = df.merge(gp, on=['device'], how='left')
del gp
df['nDiv_h'] = df['nDiv_h'].astype('uint16')
gc.collect()
print( df.info() )
