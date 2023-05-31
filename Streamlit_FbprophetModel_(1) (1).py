#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install streamlit -q')


# In[5]:


get_ipython().system('pip install --user streamlit -q')


# In[7]:


get_ipython().system('pip install streamlit')


# In[9]:


get_ipython().run_cell_magic('writefile', 'app.py', 'import streamlit as st\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nfrom prophet import Prophet\nfrom prophet.diagnostics import performance_metrics\nfrom prophet.diagnostics import cross_validation\nfrom prophet.plot import plot_cross_validation_metric\nimport base64\nfrom itertools import cycle\nimport plotly.express as px\n\nst.title(\'Reliance_Stock Forecasting of next 30days using Streamlit\')\nst.subheader(\'By Srinu Guddala\')\n\nst.write("IMPORT DATA")\nst.write("Import the time series csv file. It should have two columns labelled as \'ds\' and \'y\'.The \'ds\' column should be of datetime format by Pandas. The \'y\' column must be numeric representing the measurement to be forecasted.")\n\ndata = st.file_uploader(\'Upload here\',type=\'csv\')\n\nif data is not None:\n    new_names = [\'ds\', \'y\']\n    appdata = pd.read_csv(data,names=new_names,header=0,usecols=[0,1])\n    appdata[\'ds\'] = pd.to_datetime(appdata[\'ds\'],errors=\'coerce\') \n    \n    st.write(data)\n    \n    max_date = appdata[\'ds\'].max()\n\nst.write("SELECT FORECAST PERIOD")\n\nperiods_input = st.slider(\'How many days forecast do you want?\',min_value = 1, max_value = 30)\n             \nif data is not None:\n    obj = Prophet()\n    obj.fit(appdata)\n\nst.write("VISUALIZE FORECASTED DATA")\nst.write("The following plot shows future predicted values. \'yhat\' is the predicted value; upper and lower limits are 80% confidence intervals by default")\n\nif data is not None:\n    future = obj.make_future_dataframe(periods=periods_input)\n    \n    fcst = obj.predict(future)\n    forecast = fcst[[\'ds\', \'yhat\', \'yhat_lower\', \'yhat_upper\']]\n\n    forecast_filtered =  forecast[forecast[\'ds\'] > max_date]    \n    st.write(forecast_filtered)\n\n    \n    st.write("The next visual shows the actual (black dots) and predicted (blue line) values over time.")    \n\n    figure1 = obj.plot(fcst)\n    st.write(figure1)\n \n    \n    st.write("The next few visuals show a high level trend of predicted values, day of week trends, and yearly trends (if dataset covers multiple years). The blue shaded area represents upper and lower confidence intervals.")\n      \n\n    figure2 = obj.plot_components(fcst)\n    st.write(figure2)\n\n\n')


# # Local Tunnel concept
# 

# In[10]:


get_ipython().system('npm install -g localtunnel')


# In[8]:


get_ipython().system('streamlit run app.py & npx localtunnel --port 8501')


# In[ ]:




