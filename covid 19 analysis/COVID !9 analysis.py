#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
get_ipython().run_line_magic('matplotlib', 'inline')
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot 
init_notebook_mode(connected=True)

init_notebook_mode(connected=True)
cf.go_offline()

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from IPython.core.display import HTML



import warnings
warnings.filterwarnings('ignore')


# In[3]:



import requests


url = "https://www.worldometers.info/coronavirus/#countries"

header = {
  "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36",
  "X-Requested-With": "XMLHttpRequest"
}
r = requests.get(url, headers=header)

dfs = pd.read_html(r.text)
df = dfs[0]

time_series = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv',parse_dates=['Date'])


# In[ ]:


df.isnull().sum()


# In[4]:


# renaming the columns
df.rename(columns = {'Country,Other':'Country'},inplace = True)
df.rename(columns = {'Serious,Critical':'Serious'},inplace = True)
df.rename(columns = {'Tot Cases/1M pop':'TotalCases/1M'},inplace = True)
time_series.rename(columns = {'Country/Region':'Country'},inplace = True)
df.drop('#',axis =1,inplace = True)
df.head()


# In[5]:


#drop the columns in time_series
time_series.drop('Province/State',axis =1,inplace = True)
time_series.head()


# In[6]:


#changing the data type

df['TotalCases'] = df['TotalCases'].fillna(0).astype('int')
df['TotalDeaths'] = df['TotalDeaths'].fillna(0).astype('int')
df['TotalRecovered'] = df['TotalRecovered'].fillna(0).astype('int')
df['ActiveCases'] = df['ActiveCases'].fillna(0).astype('int')
df['Serious'] = df['Serious'].fillna(0).astype('int')
df['Deaths/1M pop'] = df['Deaths/1M pop'].fillna(0).astype('int')
df['TotalTests'] = df['TotalTests'].fillna(0).astype('int')
df['Tests/ 1M pop'] = df['Tests/ 1M pop'].fillna(0).astype('int')
df['NewCases'] = df['NewCases'].fillna(0)
df[['NewCases']] = df[['NewCases']].replace('[\+,]', '', regex=True).astype(int)
df['NewDeaths'] = df['NewDeaths'].fillna(0)
df[['NewDeaths']] = df[['NewDeaths']].replace('[\+,]', '', regex=True).astype(int)
df[['Population']] = df[['Population']].fillna(0).astype('int')
time_series.fillna(0)
time_series.isnull().sum()


# In[7]:


#highlighting the most no of cases
dataframe = df.iloc[1:216,:-1]


dataframe.style.background_gradient(cmap = 'Reds')


# In[8]:


# Heatmap confirmed Cases
group1 = time_series.groupby(['Date', 'Country'])['Confirmed', 'Deaths','Recovered'].sum().reset_index()
heat= px.choropleth(group1, locations="Country", locationmode='country names', color=np.log(group1["Confirmed"]), 
                    hover_name="Country",projection = 'natural earth',title='Heatmap', color_continuous_scale=px.colors.sequential.Blues)

heat.update(layout_coloraxis_showscale=False)
heat.show()


# In[ ]:


# heap map deaths

fig_heat= px.choropleth(group1, locations="Country", locationmode='country names', color=np.log(group1["Deaths"]), 
                    hover_name="Country",projection = 'natural earth',title='Heatmap(Deaths)', 
                    color_continuous_scale=px.colors.sequential.Reds)

fig_heat.update(layout_coloraxis_showscale=False)

fig_heat.show()


# In[ ]:


# top countries 
fig_z = px.bar(dataframe.sort_values('TotalCases'),x='TotalCases', y='Country',orientation = 'h',

            color_discrete_sequence=['#B3611A'],text = 'TotalCases',title='TotalCases')


fig_x = px.bar(dataframe.sort_values('TotalDeaths'),x='TotalDeaths', y='Country',orientation = 'h',
               color_discrete_sequence=['#830707'],text = 'TotalDeaths',title = 'TotalDeaths')


fig_ = px.bar(dataframe.sort_values('TotalRecovered'),x='TotalRecovered',y='Country',orientation ='h',
               color_discrete_sequence=['#073707'],text = 'TotalRecovered',title = 'TotalRecovered')

fig_p = make_subplots(rows =1,cols =3,subplot_titles=('TotalCases','TotalDeaths','TotalRecovered'))

fig_p.add_trace(fig_z['data'][0],row = 1,col =1)
fig_p.add_trace(fig_x['data'][0],row = 1,col =2)
fig_p.add_trace(fig_['data'][0],row=1,col=3)

fig_p.update_layout(height=3000,title ='Per Country')
fig_p.show()


# In[ ]:


# Top 20 countries mostly affected
data = totalCases.sort_values('TotalCases')
data1 = TotalDeaths.sort_values('TotalDeaths')
data2 = totalrecovered.sort_values('TotalRecovered')

fig1 = px.bar(data,x="TotalCases", y="Country",orientation = 'h',color_discrete_sequence=['#B3611A'],text='TotalCases')

fig2 = px.bar(data1,x="TotalDeaths", y="Country",orientation = 'h',color_discrete_sequence =['#830707'],text = 'TotalDeaths')

fig3 = px.bar(data2,x='TotalRecovered',y='Country',orientation = 'h',color_discrete_sequence=['#073707'],text = 'TotalRecovered')



fig = make_subplots(
    rows=2, cols=3,
    subplot_titles=("Totalconfirmed", "Total deaths", "total Recovered"))

fig.add_trace(fig1['data'][0], row=1, col=1)
fig.add_trace(fig2['data'][0], row=1, col=2)
fig.add_trace(fig3['data'][0], row=1, col=3)

fig.update_layout(height=1200,title = 'Top 20 Countries')


# In[9]:


# Confirmed vs Deaths vs Recoverd in world

grp_country1 = time_series.groupby(['Date'])['Confirmed','Deaths','Recovered'].sum().reset_index()
fig = go.Figure()
fig.add_trace(go.Scatter(x=grp_country1['Date'], y=grp_country1['Confirmed'],
                    mode='lines',
                    name='Confirmed'))
fig.add_trace(go.Scatter(x=grp_country1['Date'], y=grp_country1['Recovered'],
                    mode='lines',
                    name='Recovered',fillcolor = 'green'))
fig.add_trace(go.Scatter(x=grp_country1['Date'], y=grp_country1['Deaths'],
                    mode='lines',
                    name='Deaths',fillcolor = 'red'))
fig.update_layout(title = 'Confirmed vs Deaths vs Recoverd in world')
fig.show()


# In[ ]:


grp_country = time_series.groupby(['Date',"Country"])['Confirmed','Deaths','Recovered'].sum().reset_index()

fig_a = px.bar(grp_country, x = 'Date', y = 'Confirmed', color = 'Country',height = 500,
      title = 'Total Confirmed Cases ')

fig_a.show()

fig_b= px.bar(grp_country, x = 'Date', y = 'Deaths',color = 'Country',height = 500,
      title = 'Total Deaths')
fig_b.show()

fig_c=px.bar(grp_country,x='Date',y = 'Recovered', color= 'Country',height = 500,
      title = 'Total Recovered')
fig_c.show()


# In[ ]:


# animation of deaths world wide

group1 = time_series.groupby(['Date', 'Country'])['Confirmed', 'Deaths','Recovered'].sum().reset_index()
fig7= px.choropleth(group1, locations="Country", locationmode='country names', color=np.log(group1["Deaths"]), 
                    hover_name="Country",hover_data = ['Deaths'] ,animation_frame=group1["Date"].dt.strftime('%Y-%m-%d'),
                    projection = 'natural earth',
                    title='Deaths Over Time', color_continuous_scale=px.colors.sequential.Reds)
fig7.update(layout_coloraxis_showscale=False)
fig7.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




