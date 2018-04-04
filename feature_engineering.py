# add new features: Grouping by ['ip', 'app', 'channel'], and aggregating hour with mean
print('group by: ip_app_channel')
gp = df[['ip','app', 'channel', 'hour']].groupby(by=['ip', 'app', 'channel'])[['hour']].mean().reset_index().rename(index=str, columns={'hour': 'ip_app_channel_mean_hour'}).astype('float32')
df = df.merge(gp, on=['ip', 'app', 'channel'], how='left')
del gp
gc.collect()
print( df.info() )


print('group by: device hour')
gp = df[['device','hour']].groupby(by=['device'])[['hour']].count().reset_index().rename(index=str, columns={'hour': 'nDiv_h'})
df = df.merge(gp, on=['device'], how='left')
del gp
df['nDiv_h'] = df['nDiv_h'].astype('uint16')
gc.collect()
print( df.info() )

# # of clicks for each ip-app-channel combination
print("group by # of clicks for each ip-app-channel combination")
gp = df[['ip','app', 'os', 'channel']].groupby(by=['ip', 'app', 'channel'])[['os']].count().reset_index().rename(index=str, columns={'os': 'ip_app_channel_count'})
df = df.merge(gp, on=['ip','app', 'channel'], how='left')
del gp
df['ip_app_channel_count'] = df['ip_app_channel_count'].astype('uint16')
gc.collect()



print("group by channel_count...")
gp = train_df[['app', 'channel']].groupby(by=['channel'])[['app']].count().reset_index().rename(index=str, columns={'app': 'channel_count'})
train_df = train_df.merge(gp, on=['channel'], how='left')
del gp
df['channel_count'] = df['channel_count'].astype('uint16')
gc.collect()

print("group by ip_app_device_count...")
gp = train_df[['ip','app', 'device', 'channel']].groupby(by=['ip', 'app', 'device'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_device_count'})
train_df = train_df.merge(gp, on=['ip','app', 'device'], how='left')
del gp
df['ip_app_device_count'] = df['ip_app_device_count'].astype('uint16')
gc.collect()



