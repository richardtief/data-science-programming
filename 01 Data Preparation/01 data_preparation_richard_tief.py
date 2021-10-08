
# coding: utf-8

# In[1]:


import sqlite3
import pandas as pd


# In[2]:


# connection to database
connection = sqlite3.connect('the.db')
cursor = connection.cursor()


# In[3]:


# explore data
cursor.execute('SELECT * FROM dwd')
data = cursor.fetchall()
for row in data:
    print(row)


# In[4]:


# explore data with pandas data frame
df = pd.read_sql_query("SELECT * FROM dwd", connection)
df


# In[5]:


# get latest timestamp
latest_timestamp = df['timestamp'].max()
latest_timestamp


# In[6]:


# download file from url
target_file = 'https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/air_temperature/recent/10minutenwerte_TU_01048_akt.zip'

fetched_data = pd.read_csv(target_file, sep=';')
fetched_data


# In[7]:


# explore data types
fetched_data.info()


# In[8]:


# align data with current database and set desired range
fetched_data.drop(['eor','  QN'], axis=1, inplace=True)
fetched_data.rename(columns={
        'STATIONS_ID': 'id',
        'MESS_DATUM': 'timestamp',
        'PP_10': 'airpressure',
        'TT_10': 'temperature',
        'TM5_10': 'temperature_ground',
        'RF_10': 'humidity',
        'TD_10': 'temperature_dew'
}, 
                    inplace=True)
fetched_data = fetched_data.astype({'timestamp': str})
fetched_data = fetched_data[fetched_data['timestamp'] > latest_timestamp]
fetched_data = fetched_data[fetched_data['timestamp'] < '202010230000']
fetched_data


# In[9]:


# reset index
updated_data = pd.concat([df, fetched_data]).reset_index()
updated_data.drop(['index'], axis=1, inplace=True)
updated_data


# In[10]:


# adding new data to database
fetched_data.to_sql('dwd', connection, if_exists='append', index=False)


# In[11]:


# export database as json file
updated_data.to_json('weather_data.json')


# In[12]:


# convert timestamp column into datetime format
updated_data['timestamp'] = pd.to_datetime(updated_data['timestamp'])


# In[13]:


# create table with averaged data per hour
averaged_data = updated_data.groupby(pd.DatetimeIndex(updated_data.timestamp).hour).mean()
averaged_data.index.names = ['hour']
averaged_data


# In[14]:


# writing data to database
averaged_data.to_sql('averaged_data', connection, if_exists='replace')


# In[15]:


# check if it is there
df1 = pd.read_sql_query("SELECT * FROM averaged_data", connection)
df1


# In[16]:


# close connection to database
connection.close()

