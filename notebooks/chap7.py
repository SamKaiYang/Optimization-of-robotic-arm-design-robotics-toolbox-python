#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().magic(u'matplotlib widget')
import roboticstoolbox as rp
import numpy as np

panda = rp.models.DH.Panda()


# In[2]:


print(panda.fkine())


# In[3]:


panda.teach(block=True)


# In[5]:


panda.plot(block=False)


# In[ ]:




