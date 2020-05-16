#!/usr/bin/env python
# coding: utf-8

# In[30]:


import turicreate


# ## Read some product review data
# 

# In[31]:


products = turicreate.SFrame('amazon_baby.sframe')


# In[32]:


products


# In[33]:


products.head()


# In[34]:


products['word_count']=turicreate.text_analytics.count_words(products['review'])


# In[35]:


products


# In[36]:


turicreate.set_target('ipynb')


# In[37]:


products['name'].show()


# In[38]:


giraffe_reviews= products[products['name'] =='Vulli Sophie the Giraffe Teether']


# In[39]:


len(giraffe_reviews)


# In[40]:


giraffe_reviews['rating'].show()


# In[41]:


products['rating'].show()


# In[42]:


# ignore all 3* reviews


# In[43]:


products=products[products['rating'] !=3]


# In[44]:


products['sentiment']=products['rating']>=4


# In[45]:


products


# In[46]:


train_data,test_data = products.random_split(0.8,seed=0)


# In[60]:


sentiment_model=turicreate.logistic_classifier.create(train_data,target='sentiment',features=['word_count'],validation_set=test_data,)


# In[48]:


roc_eval=sentiment_model.evaluate(test_data,metric='roc_curve')


# In[49]:


sentiment_model.roc_curve


# In[ ]:


import matplotlib.pyplot as plt
#%matplotlib inline

plt.scatter(roc_eval['roc_curve']['fpr'],
            roc_eval['roc_curve']['tpr'],  
            label='ROC Curve')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
#%matplotlib inline

plt.scatter(roc_eval['roc_curve']['fpr'],
            roc_eval['roc_curve']['tpr'],  
            label='ROC Curve')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()


# In[52]:


giraffe_reviews[0]['review']


# In[ ]:


giraffe_reviews[1]['review']


# In[ ]:


giraffe_reviews[2]['review']


# In[ ]:


giraffe_reviews[3]['review']


# In[55]:


giraffe_reviews['predicted_sentiment']=sentiment_model.predict(giraffe_reviews,output_type='probability')


# In[57]:


giraffe_reviews[0]['review']


# In[59]:


giraffe_reviews[1]['review']


# In[ ]:





# In[ ]:




