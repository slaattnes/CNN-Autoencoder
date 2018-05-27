
# coding: utf-8

# In[1]:


import pickle
from get_actions import initialize, add_frame
import numpy as np


# In[2]:


windowed_features = pickle.load(open('./cluster_data.p', "rb"))


# In[3]:


initialize(actions=8)


# In[5]:


for i in range(1000):
    c = add_frame(np.array([1,2,3,4]))
    if (i % 10 == 0 and c is not -1):
        #print (i)
        
        print ("ACTION: ", c)

