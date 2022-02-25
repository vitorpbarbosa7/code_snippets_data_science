#!/usr/bin/env python
# coding: utf-8

# In[16]:


root = '/media/vpb/GD_/USP/DS/01Git/01_scripts_functions_pipelines/28_papermill/'


# In[17]:


# ! python3 -m pip install papermill


# In[18]:


import datetime
now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')


# In[19]:


with open(root+now, 'w') as f:
    f.write(now)

