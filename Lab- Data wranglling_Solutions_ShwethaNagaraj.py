#!/usr/bin/env python
# coding: utf-8

# In[4]:


#1. Import pandas under the name pd.

import pandas as pd


# In[3]:


#2. Print the version of pandas that has been imported.

pd.__version__


# In[5]:


#3. Print out all the version information of the libraries that are required by the pandas library.

pd.show_versions()


# In[16]:


import numpy as np
data = {'animal': ['cat', 'cat', 'snake', 'dog', 'dog', 'cat', 'snake', 'cat', 'dog', 'dog'],
        'age': [2.5, 3, 0.5, np.nan, 5, 2, 4.5, np.nan, 7, 3],
        'visits': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'priority': ['yes', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no']}

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']


# In[17]:


#4.Create a DataFrame df from this dictionary data which has the index labels.

df = pd.DataFrame(data, index=labels)


# In[18]:


#5. Display a summary of the basic information about this DataFrame and its data.

df.describe()


# In[19]:


#6. Return the first 3 rows of the DataFrame df.

df.head(3)


# In[20]:


#7. Select just the 'animal' and 'age' columns from the DataFrame df.

df[['animal', 'age']]


# In[21]:


#8. Select the data in rows [3, 4, 8] and in columns ['animal', 'age'].

df.loc[df.index[[3, 4, 8]], ['animal', 'age']]


# In[ ]:


#9. Select only the rows where the number of visits is greater than 3.

df[df['visits'] > 3]


# In[22]:


#10. Select the rows where the age is missing, i.e. is NaN.

df[df['age'].isnull()]


# In[23]:



#11. Select the rows where the animal is a cat and the age is less than 3.

df[(df['animal'] == 'cat') & (df['age'] < 3)]


# In[25]:


#12. Select the rows the age is between 2 and 4 (inclusive).

df[df['age'].between(2, 4)]


# In[27]:


#13. Change the age in row 'f' to 1.5.

df.loc['f', 'age'] = 1.5


# In[28]:



#14. Calculate the sum of all visits (the total number of visits).

df['visits'].sum()


# In[29]:



#15. Calculate the mean age for each different animal in df.

df.groupby('animal')['age'].mean()


# In[53]:




#16. Append a new row 'k' to df with your choice of values for each column. Then delete that row to return the original DataFrame.

df.loc['k'] = [5.5, 'dog', 'no', 2]
df.describe


# In[54]:


#16 Deleting the new row:

df = df.drop('k')
df.describe


# In[55]:



#17. Count the number of each type of animal in df.

df['animal'].value_counts()


# In[56]:


#18. Sort df first by the values in the 'age' in decending order, then by the value in the 'visit' column in ascending order.

df.sort_values(by=['age', 'visits'], ascending=[False, True])


# In[57]:


#19. The 'priority' column contains the values 'yes' and 'no'. Replace this column with a column of boolean values: 'yes' should be True and 'no' should be False.

df['priority'] = df['priority'].map({'yes': True, 'no': False})


# In[58]:


#20. In the 'animal' column, change the 'snake' entries to 'python'.

df['animal'] = df['animal'].replace('snake', 'python')


# In[76]:


#22.You have a DataFrame df with a column 'A' of integers. For example:
#How do you filter out rows which contain the same integer as the row immediately above?

df = pd.DataFrame({'A': [1, 2, 2, 3, 4, 5, 5, 5, 6, 7, 7]})

df.loc[df['A'].shift() != df['A']]


# In[64]:


#23. Given a DataFrame of numeric values, say # a 5x3 frame of float values
#how do you subtract the row mean from each element in the row?

df = pd.DataFrame(np.random.random(size=(5, 3)))

df.sub(df.mean(axis=1), axis=0)


# In[65]:


#24. Suppose you have DataFrame with 10 columns of real numbers, for example:
#Which column of numbers has the smallest sum? (Find that column's label.)

df = pd.DataFrame(np.random.random(size=(5, 10)), columns=list('abcdefghij'))

df.sum().idxmin()


# In[75]:


#25. How do you count how many unique rows a DataFrame has (i.e. ignore all rows that are duplicates)?

len(df.drop_duplicates(keep=False))


# In[66]:


#26. You have a DataFrame that consists of 10 columns of floating--point numbers. Suppose that exactly 5 entries in each row are NaN values. For each row of the DataFrame, find the column which contains the third NaN value.
#(You should return a Series of column labels.)

(df.isnull().cumsum(axis=1) == 3).idxmax(axis=1)


# In[67]:


#27. A DataFrame has a column of groups 'grps' and and column of numbers 'vals'. For example:
#For each group, find the sum of the three greatest values.

df = pd.DataFrame({'grps': list('aaabbcaabcccbbc'), 
                   'vals': [12,345,3,1,45,14,4,52,54,23,235,21,57,3,87]})

df.groupby('grps')['vals'].nlargest(3).sum(level=0)


# In[74]:


#29. Consider a DataFrame df where there is an integer column 'X':
#For each value, count the difference back to the previous zero (or the start of the Series, whichever is closer). 
#These values should therefore be [1, 2, 0, 1, 2, 3, 4, 0, 1, 2]. 
#Make this a new column 'Y'.

df = pd.DataFrame({'X': [7, 2, 0, 3, 4, 2, 5, 0, 3, 4]})

df['Y'] = df.groupby((df['X'] == 0).cumsum()).cumcount()
# We're off by one before we reach the first zero.
first_zero_idx = (df['X'] == 0).idxmax()
df['Y'].iloc[0:first_zero_idx] += 1

df


# In[77]:


#30. Consider a DataFrame containing rows and columns of purely numerical data. Create a list of the row-column index locations of the 3 largest values.

df.unstack().sort_values()[-3:].index.tolist()


# In[88]:


#32. Implement a rolling mean over groups with window size 3, which ignores NaN value. For example consider the following DataFrame:
#Goal is to compute a series. E.g. the first window of size three for group 'b' has values 3.0, NaN and 3.0 and occurs at row index 5. Instead of being NaN the value in the new column at this row index should be 3.0 (just the two non-NaN values are used to compute the mean (3+3)/2)
df = pd.DataFrame({'group': list('aabbabbbabab'),
                       'value': [1, 2, 3, np.nan, 2, 3, 
                                 np.nan, 1, 7, 3, np.nan, 8]})
df


# In[89]:


#32 continued:
g1 = df.groupby(['group'])['value']              # group values  
g2 = df.fillna(0).groupby(['group'])['value']    # fillna, then group values

s = g2.rolling(3, min_periods=1).sum() / g1.rolling(3, min_periods=1).count() # compute means

s.reset_index(level=0, drop=True).sort_index()  # drop/sort index


# In[95]:


#33. Create a DatetimeIndex that contains each business day of 2015 and use it to index a Series of random numbers. Let's call this Series s.

dti = pd.date_range(start='2015-01-01', end='2015-12-31', freq='B') 
s = pd.Series(np.random.rand(len(dti)), index=dti)

s


# In[96]:


#34. Find the sum of the values in s for every Wednesday.

s[s.index.weekday == 2].sum()


# In[97]:


#35. For each calendar month in s, find the mean of values.


s.resample('M').mean()


# In[99]:



#37. Create a DateTimeIndex consisting of the third Thursday in each month for the years 2015 and 2016.

pd.date_range('2015-01-01', '2016-12-31', freq='WOM-3THU')


# In[100]:


df = pd.DataFrame({'From_To': ['LoNDon_paris', 'MAdrid_miLAN', 'londON_StockhOlm', 
                               'Budapest_PaRis', 'Brussels_londOn'],
              'FlightNumber': [10045, np.nan, 10065, np.nan, 10085],
              'RecentDelays': [[23, 47], [], [24, 43, 87], [13], [67, 32]],
                   'Airline': ['KLM(!)', '<Air France> (12)', '(British Airways. )', 
                               '12. Air France', '"Swiss Air"']})


# In[103]:


#38. Some values in the the FlightNumber column are missing. These numbers are meant to increase by 10 with each row so 10055 and 10075 need to be put in place. Fill in these missing numbers and make the column an integer column (instead of a float column).

df['FlightNumber'] = df['FlightNumber'].interpolate().astype(int)
df


# In[106]:


#39. The From_To column would be better as two separate columns! Split each string on the underscore delimiter _ to give a new temporary DataFrame with the correct values. Assign the correct column names to this temporary DataFrame.

temp = df.From_To.str.split('_', expand=True)
temp.columns = ['From', 'To']

temp


# In[107]:


#40. Notice how the capitalisation of the city names is all mixed up in this temporary DataFrame. Standardise the strings so that only the first letter is uppercase (e.g. "londON" should become "London".)


temp['From'] = temp['From'].str.capitalize()
temp['To'] = temp['To'].str.capitalize()

temp


# In[108]:


#41. Delete the From_To column from df and attach the temporary DataFrame from the previous questions.


df = df.drop('From_To', axis=1)
df = df.join(temp)

df


# In[110]:


#42. In the Airline column, you can see some extra puctuation and symbols have appeared around the airline names. Pull out just the airline name. E.g. '(British Airways. )' should become 'British Airways'.

df['Airline'] = df['Airline'].str.extract('([a-zA-Z\s]+)', expand=False).str.strip() #Also,strip() removes any leading/trailing spaces
df


# In[111]:


#43. In the RecentDelays column, the values have been entered into the DataFrame as a list. We would like each first value in its own column, each second value in its own column, and so on. If there isn't an Nth value, the value should be NaN.
#Expand the Series of lists into a DataFrame named delays, rename the columns delay_1, delay_2, etc. and replace the unwanted RecentDelays column in df with delays.
# there are several ways to do this, but the following approach is possibly the simplest

delays = df['RecentDelays'].apply(pd.Series)

delays.columns = ['delay_{}'.format(n) for n in range(1, len(delays.columns)+1)]

df = df.drop('RecentDelays', axis=1).join(delays)

df


# In[112]:


#44. Given the lists letters = ['A', 'B', 'C'] and numbers = list(range(10)), construct a MultiIndex object from the product of the two lists. Use it to index a Series of random numbers. Call this Series s.

letters = ['A', 'B', 'C']
numbers = list(range(10))

mi = pd.MultiIndex.from_product([letters, numbers])
s = pd.Series(np.random.rand(30), index=mi)

s


# In[113]:


#45. Check the index of s is lexicographically sorted (this is a necessary proprty for indexing to work correctly with a MultiIndex).

s.index.lexsort_depth == s.index.nlevels


# In[114]:


#46. Select the labels 1, 3 and 6 from the second level of the MultiIndexed Series.

s.loc[:, [1, 3, 6]]


# In[115]:


#47. Slice the Series s; slice up to label 'B' for the first level and from label 5 onwards for the second level.

s.loc[slice(None, 'B'), slice(5, None)]


# In[116]:


#48. Sum the values in s for each label in the first level (you should have Series giving you a total for labels A, B and C).

s.sum(level=0)


# In[117]:


#49. Suppose that sum() (and other methods) did not accept a level keyword argument. How else could you perform the equivalent of s.sum(level=1)?

s.unstack().sum(axis=0)


# In[124]:


#50. Exchange the levels of the MultiIndex so we have an index of the form (letters, numbers). Is this new Series properly lexsorted? If not, sort it.

new_s = s.swaplevel(0, 1)
# check
new_s.index.is_lexsorted() #it is not sorted.


new_s



# In[125]:


#50 continued

# sort
new_s = new_s.sort_index()
new_s


# In[126]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')


# In[128]:


#56. Pandas is highly integrated with the plotting library matplotlib, and makes plotting DataFrames very user-friendly! Plotting in a notebook environment usually makes use of the following boilerplate:
df = pd.DataFrame({"xs":[1,5,2,8,1], "ys":[4,2,1,9,6]})
df.plot.scatter("xs", "ys", color = "black", marker = "x")


# In[132]:


#57. Columns in your DataFrame can also be used to modify colors and sizes. Bill has been keeping track of his performance at work over time, as well as how good he was feeling that day, and whether he had a cup of coffee in the morning. Make a plot which incorporates all four features of this DataFrame.


df = pd.DataFrame({"productivity":[5,2,3,1,4,5,6,7,8,3,4,8,9],
                   "hours_in"    :[1,9,6,5,3,9,2,9,1,7,4,2,2],
                   "happiness"   :[2,1,3,2,3,1,2,3,1,2,2,1,3],
                   "caffienated" :[0,0,1,1,0,0,0,0,1,1,0,1,0]})

df.plot.scatter("hours_in", "productivity", s = df.happiness * 30, c = df.caffienated)


# In[133]:


#58. What if we want to plot multiple things? Pandas allows you to pass in a matplotlib Axis object for plots, and plots will also return an Axis object.
#Make a bar plot of monthly revenue with a line plot of monthly advertising spending (numbers in millions)

df = pd.DataFrame({"revenue":[57,68,63,71,72,90,80,62,59,51,47,52],
                   "advertising":[2.1,1.9,2.7,3.0,3.6,3.2,2.7,2.4,1.8,1.6,1.3,1.9],
                   "month":range(12)
                  })

ax = df.plot.bar("month", "revenue", color = "green")
df.plot.line("month", "advertising", secondary_y = True, ax = ax)
ax.set_xlim((-1,12))


# In[134]:


#This function is designed to create semi-interesting random stock price data

import numpy as np
def float_to_time(x):
    return str(int(x)) + ":" + str(int(x%1 * 60)).zfill(2) + ":" + str(int(x*60 % 1 * 60)).zfill(2)

def day_stock_data():
    #NYSE is open from 9:30 to 4:00
    time = 9.5
    price = 100
    results = [(float_to_time(time), price)]
    while time < 16:
        elapsed = np.random.exponential(.001)
        time += elapsed
        if time > 16:
            break
        price_diff = np.random.uniform(.999, 1.001)
        price *= price_diff
        results.append((float_to_time(time), price))
    
    
    df = pd.DataFrame(results, columns = ['time','price'])
    df.time = pd.to_datetime(df.time)
    return df

def plot_candlestick(agg):
    fig, ax = plt.subplots()
    for time in agg.index:
        ax.plot([time.hour] * 2, agg.loc[time, ["high","low"]].values, color = "black")
        ax.plot([time.hour] * 2, agg.loc[time, ["open","close"]].values, color = agg.loc[time, "color"], linewidth = 10)

    ax.set_xlim((8,16))
    ax.set_ylabel("Price")
    ax.set_xlabel("Hour")
    ax.set_title("OHLC of Stock Value During Trading Day")
    plt.show()


# In[135]:


#59. Generate a day's worth of random stock data, and aggregate / reformat it so that it has hourly summaries of the opening, highest, lowest, and closing prices

df = day_stock_data()
df.head()


# In[138]:


#59 codf.set_index("time", inplace = True)

df.set_index("time", inplace = True)
agg = df.resample("H").ohlc()
agg.columns = agg.columns.droplevel()
agg["color"] = (agg.close > agg.open).map({True:"green",False:"red"})
agg.head()


# In[139]:


plot_candlestick(agg)

