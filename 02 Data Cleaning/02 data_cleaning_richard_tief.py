
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# explore data
df = pd.read_csv('data-cleaning.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')
df


# In[3]:


lower = df.min().temperature
upper = df.max().temperature
print("Without filter:")
print("Boundaries:", lower, "> IQR <", upper)
print("Range:", upper - lower)


# In[4]:


# plot data to get an overview
plt.boxplot(df['temperature'])
plt.show()


# In[5]:


# IQR Filter
# Remove outliers from a dataframe by column.
# Removing rows for which the column value are
# less than Q1-1.5IQR or greater than Q3+1.5IQR.
def iqr_range_filter(dataframe, column):
    """
    Args:
        df (`:obj:pd.DataFrame`): A pandas dataframe to subset
        column (str): Name of the column to calculate the subset from.
    Returns:
        (`:obj:pd.DataFrame`): Filtered dataframe
    """
    # Calculate Q1, Q2 and IQR
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    # Apply filter with respect to IQR
    filter = (dataframe[column] >= q1 - 1.5*iqr) and (dataframe[column] <= q3 + 1.5*iqr)
    return df.loc[filter]

# Apply function
df_filtered_by_iqr = iqr_range_filter(df, 'temperature')
lower = df_filtered_by_iqr.min().temperature
upper = df_filtered_by_iqr.max().temperature
print("IQR boundaries:", lower, "> IQR <", upper)
print("IQR range:", upper - lower)

# filtered dataframe
df_filtered_by_iqr = iqr_range_filter(df, 'temperature')
df_filtered_by_iqr


# In[6]:


# plot data after IQR filter
plt.boxplot(df_filtered_by_iqr['temperature'])
plt.show()


# In[7]:


# Z-Score Filter
# Remove outliers from a dataframe by column.
# Removing rows for which the column value are
# less than -3 or greater than 3 in z-standardized data.
#
def z_score_range_filter(dataframe, column):
    """
    Args:
        df (`:obj:pd.DataFrame`): A pandas dataframe to subset
        column (str): Name of the column to calculate the subset from.
    Returns:
        (`:obj:pd.DataFrame`): Filtered dataframe
    """
    mean = dataframe[column].mean()
    std = dataframe[column].std()
    lower = mean - 3 * std
    upper = mean + 3 * std
    # Apply filter with respect to IQR
    filter = (dataframe[column] >= lower) & (dataframe[column] <= upper)
    return df.loc[filter]

# Apply function
df_filtered_by_z_score = z_score_range_filter(df, 'temperature')
lower = df_filtered_by_z_score.min().temperature
upper = df_filtered_by_z_score.max().temperature
print("Z-Score boundaries:", lower, "> mue <", upper)
print("Z-Score range:", upper - lower)

# filtered dataframe
df_filtered_by_z_score = z_score_range_filter(df, 'temperature')
df_filtered_by_z_score


# In[8]:


# plot data after IQR filter
plt.boxplot(df_filtered_by_z_score['temperature'])
plt.show()


# In[9]:


# Replace outliers with NA values
df['temperature'] = df['temperature'].where(df['temperature'].between(lower, upper))
print(df["temperature"].isna().sum(), "outliers replaced with NaN.")


# In[10]:


# Fill missing data points with NA's
length = len(df['temperature'])
# resample dataframe
df_with_nans = df.resample('10Min').asfreq()
number_of_gaps = len(df_with_nans["temperature"]) - length
print(number_of_gaps, "gaps filled with NaN")
#show NaN entrys
df_with_nans[df_with_nans.isna().any(axis=1)]


# In[11]:


# explore the gap locations
gaps = np.where(df_with_nans['temperature'].isna())[0]
split = np.where(np.diff(gaps) > 1)[0] + 1
gaps = np.split(gaps, split)
gaps


# In[12]:


for gap in gaps:
    print(df_with_nans.iloc[gap])


# In[13]:


def linear_interpolation(start, end, length):
    return np.linspace(start, end, length)

def step_interpolation(start, end, length):
    l = length //2
    return [start]*l + [end]*l

def interpolate(dataframe, column, function):
    gaps = np.where(dataframe[column].isna())[0]
    split = np.where(np.diff(gaps) > 1)[0] + 1
    gaps = np.split(gaps, split)
    l = 0

    for gap in gaps:
        length = len(gap)
        start = gap[0] -1
        end = gap[-1] + 1
        last_value = dataframe.iloc[gap[0] -1][0]
        first_value_after_gap = dataframe.iloc[gap[-1] + 1][0]
        l += length
        if length < 2:
            dataframe.iloc[gap[0]] = ((first_value_after_gap + last_value)/2)
            print("\n \n Interpolated value linear between:",
                  last_value, "and",
                  first_value_after_gap,
                  "| number values:", length)
            print(dataframe.iloc[gap])
        else:
            dataframe.iloc[start+1:end,0] = function(last_value, first_value_after_gap, length)
            print("\n \n Interpolated values linear between:",
                  last_value, "and",
                  first_value_after_gap,
                  "| number values:", length)
            print(dataframe.iloc[gap])
    print('\n',l , 'interpolated values.')
    return dataframe


# In[14]:


df2 = pd.DataFrame(df_with_nans)
df_interpolated = interpolate(df2, 'temperature', step_interpolation)
