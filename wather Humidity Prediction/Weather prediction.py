#!/usr/bin/env python
# coding: utf-8

# In[633]:


import numpy as np 
import pandas as pd


# In[634]:


df=pd.read_csv('daily_weather.csv')


# In[635]:


df.head(3)


# In[636]:


df.tail()


# In[637]:


df.shape


# In[638]:


df.isnull().sum()


# In[639]:


df.dropna(inplace=True)


# In[640]:


df.isnull().sum()


# In[641]:


df.head(2)


# In[642]:


df.drop(columns='number',inplace=True)


# In[643]:


df


# In[644]:


df.info()


# In[645]:


df.describe()


# In[646]:


pie=df['air_pressure_9am'].value_counts().head(10)


# In[647]:


import matplotlib.pyplot as plt
plt.pie(pie,autopct='%1.1f%%')
plt.title('Top 10 air_pressure_9am in percentage')


# In[648]:


pie


# In[649]:


pie2=df['air_pressure_9am'].value_counts().tail(10)


# In[650]:


plt.pie(pie2,autopct='%1.1f%%')
plt.title('Last 10 air_pressure_ in percentage')


# In[651]:


df.head()


# In[652]:


import seaborn as sns
plt.figure(figsize=(15,8))
sns.heatmap(df.corr(),annot=True,cmap='rainbow')


# In[ ]:


df.info()


# In[653]:


x=df.drop(columns=['relative_humidity_3pm'],axis=1)


# In[654]:


y=df['relative_humidity_3pm']


# In[655]:


from sklearn.model_selection import train_test_split


# In[656]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[657]:


df.shape


# In[658]:


x_train.shape,x_test.shape,y_train.shape,y_test.shape


# In[659]:


from sklearn.tree import DecisionTreeRegressor


# In[660]:


model= DecisionTreeRegressor()


# In[661]:


model.fit(x_train,y_train)


# In[662]:


model.predict(x_test)


# In[663]:


model.score(x_test,y_test)


# In[664]:


df.head()


# In[665]:


x=df.drop(columns=['relative_humidity_3pm','max_wind_speed_9am','rain_accumulation_9am','avg_wind_speed_9am'],axis=1)
y=df['relative_humidity_3pm']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[699]:


mode_r1= DecisionTreeRegressor(max_leaf_nodes=10)


# In[700]:


mode_r1.fit(x_train,y_train)


# In[701]:


mode_r1.score(x_test,y_test)


# In[702]:


from sklearn.preprocessing import StandardScaler


# In[703]:


scale=StandardScaler()


# In[704]:


scale.fit(x_train,y_train)


# In[705]:


x_test_scale=scale.fit_transform(x_test)
x_train_scale=scale.fit_transform(x_train)


# In[706]:


mode_r1.fit(x_train_scale,y_train)


# In[707]:


mode_r1.score(x_test_scale,y_test)


# In[708]:


from sklearn.ensemble import RandomForestRegressor


# In[709]:


model2=RandomForestRegressor()


# In[710]:


model2.fit(x_train,y_train)


# In[712]:


model2.score(x_test_scale,y_test)


# In[711]:


model2.fit(x_train_scale,y_train)


# In[713]:


model2.score(x_test_scale,y_test)


# In[715]:


from sklearn.linear_model import LinearRegression


# In[716]:


model3=LinearRegression()


# In[717]:


model3.fit(x_train,y_train)


# In[718]:


model3.score(x_test,y_test)


# In[719]:


model3.fit(x_train_scale,y_train)


# In[720]:


model3.score(x_test_scale,y_test)


# In[721]:


from sklearn.preprocessing import PolynomialFeatures


# In[722]:


from sklearn.pipeline import make_pipeline


# In[723]:


model_0 = make_pipeline(PolynomialFeatures(degree=1), LinearRegression())


# In[724]:


model_0.fit(x_train, y_train)


# In[725]:


model_0.score(x_test, y_test)


# In[726]:


model_01 = make_pipeline(PolynomialFeatures(degree=2), DecisionTreeRegressor())


# In[727]:


model_01.fit(x_train, y_train)


# In[728]:


model_01.score(x_test, y_test)


# In[760]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('kbest', SelectKBest(k=6)),
    ('rf', RandomForestRegressor(n_estimators=10))
])




# In[761]:


pipe.fit(x_train, y_train)


# In[762]:


pipe.score(x_test, y_test)


# In[763]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('kbest', SelectKBest(k=6)),
    ('rf',LinearRegression())
])


# In[764]:


pipe.fit(x_train, y_train)


# In[765]:


pipe.score(x_test, y_test)


# In[873]:


from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

# Define the pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestRegressor()),
])

# Define the parameter grid to search over
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [5, 10, 15, 20],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
}

# Define the RandomizedSearchCV object
search = RandomizedSearchCV(
    pipeline,
    param_grid,
    n_iter=2,
    cv=2,
    verbose=5,
    n_jobs=-1
)



# In[874]:


search.fit(x_train, y_train)


# In[881]:


search.score(x_test, y_test)


# In[900]:


humidity_3pm = df.relative_humidity_3pm[:213]


# In[901]:


plt.bar(humidity_3pm, bar)


# In[892]:


df.head(2)


# In[910]:


first=df.relative_humidity_9am.head(20)
second=df.relative_humidity_3pm.head(20)
sns.lineplot(x=first,y=second)


# In[933]:


import seaborn as sns
import matplotlib.pyplot as plt

first = df.relative_humidity_9am.head(20)
second = df.relative_humidity_3pm.head(20)

# Create a DataFrame with the values and corresponding time points
data = pd.DataFrame({'9am': first, '3pm': second})

# Reset the index of the DataFrame
data = data.reset_index()

# Melt the DataFrame to convert it from wide to long format
data = data.melt('index', var_name='Time', value_name='Relative Humidity')

# Plot the line plot
sns.lineplot(data=data, x='index', y='Relative Humidity', hue='Time')



plt.ylabel('Relative Humidity')
plt.title('Comparison of Relative Humidity at 9am and 3pm')
plt.show()



# In[927]:


columns=['relative_humidity_3pm','relative_humidity_9am']


# In[928]:


df[columns]


# In[915]:


df.head(2)


# In[924]:


df.columns


# In[935]:


pred=search.predict(x_test)


# In[936]:


pred


# In[938]:


pred.shape


# In[940]:


y_test.shape


# In[949]:


df.describe()


# In[962]:


import seaborn as sns
import matplotlib.pyplot as plt

columns = ['relative_humidity_9am', 'relative_humidity_3pm']


fig, axes = plt.subplots(nrows=1, ncols=len(columns), figsize=(10, 5))


for i, column in enumerate(columns):
    sns.boxplot(data=df[column], ax=axes[i])
    axes[i].set_ylabel('Value')
    axes[i].set_title(f'Box Plot - {column}')






# In[964]:


df.head(2)


# In[965]:


import seaborn as sns
import matplotlib.pyplot as plt

columns = ['avg_wind_direction_9am', 'avg_wind_speed_9am']


fig, axes = plt.subplots(nrows=1, ncols=len(columns), figsize=(10, 5))


for i, column in enumerate(columns):
    sns.boxplot(data=df[column], ax=axes[i])
    axes[i].set_ylabel('Value')
    axes[i].set_title(f'Box Plot - {column}')


# In[966]:


import pandas as pd
import numpy as np

# Assuming you have a DataFrame called 'df' with multiple columns

# Define a function to detect outliers using Tukey's fences method
def detect_outliers(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    return outliers

# Iterate over each column and find outliers
for column in df.columns:
    outliers = detect_outliers(df[column])
    
    if outliers.empty:
        print(f"No outliers found in column: {column}")
    else:
        print(f"Outliers found in column: {column}")
        print(outliers)
    print()


# In[967]:


df.head(2)


# In[971]:


df.columns


# In[979]:


df.describe()


# In[1000]:


import seaborn as sns
import matplotlib.pyplot as plt

air_pressure_9am = df[df['air_pressure_9am'] > 907]['air_pressure_9am']
air_temp_9am = df[df['air_temp_9am'] < 50]['air_temp_9am']
avg_wind_direction_9am = df[df['avg_wind_direction_9am'] < 200]['avg_wind_direction_9am']
avg_wind_speed_9am = df[df['avg_wind_speed_9am'] < 15]['avg_wind_speed_9am']
max_wind_direction_9am = df[df['max_wind_direction_9am'] < 200]['max_wind_direction_9am']

show = [air_pressure_9am, air_temp_9am, avg_wind_direction_9am, avg_wind_speed_9am, max_wind_direction_9am]
variables = ['air_pressure_9am', 'air_temp_9am', 'avg_wind_direction_9am', 'avg_wind_speed_9am', 'max_wind_direction_9am']

plt.figure(figsize=(15, 30))
for i, col in enumerate(show):
    plt.subplot(5, 2, i + 1)
    sns.histplot(x=col)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {variables[i]}')



# In[ ]:





# In[ ]:




