#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


loan_df = pd.read_csv('D:/FILE/Data Scientist/Rakamin Data Scientist/Final Task/loan_data_2007_2014.csv')


# In[4]:


loan_df.shape


# In[5]:


loan_df.head()


# In[6]:


loan_df.info()


# In[7]:


loan_df.describe()


# In[8]:


duplicated_rows_loan_df = loan_df[loan_df.duplicated()]
print("number of duplicate rows: ", duplicated_rows_loan_df.shape)


# In[9]:


loan_df.isna().sum().sort_values(ascending = False).head(40)


# In[10]:


loan_df.drop(['Unnamed: 0','inq_last_12m','dti_joint','annual_inc_joint','total_cu_tl','open_acc_6m','open_il_6m',
              'open_il_12m','open_il_24m','mths_since_rcnt_il','total_bal_il','il_util','open_rv_12m','open_rv_24m',
              'max_bal_bc','all_util','inq_fi','verification_status_joint','mths_since_last_record',
              'mths_since_last_major_derog','desc','mths_since_last_delinq','next_pymnt_d'],axis=1,inplace=True)


# In[11]:


loan_df.isna().sum().sort_values(ascending = False).head(20)


# In[12]:


loan_df = loan_df.dropna(subset=['total_rev_hi_lim', 'tot_coll_amt','tot_cur_bal','emp_title','emp_length','last_pymnt_d',
                                 'revol_util','collections_12_mths_ex_med','last_credit_pull_d','total_acc','delinq_2yrs',
                                 'inq_last_6mths','open_acc','pub_rec','earliest_cr_line','acc_now_delinq','title','annual_inc'])


# In[13]:


loan_df.isna().sum().sort_values(ascending = False).head()


# In[14]:


loan_df.shape


# In[15]:


loan_df.info()


# In[16]:


print("\nterm:", loan_df['term'].unique())
print("\ngrade:", loan_df['grade'].unique())
print("\nsub_grade:", loan_df['sub_grade'].unique())
print("\nemp_length:", loan_df['emp_length'].unique())
print("\nhome_ownership:", loan_df['home_ownership'].unique())
print("\nverification_status:", loan_df['verification_status'].unique())
print("\nloan_status:", loan_df['loan_status'].unique())
print("\npymnt_plan:", loan_df['pymnt_plan'].unique())
print("\npurpose:", loan_df['purpose'].unique())
print("\ntitle:", loan_df['title'].unique())
print("\naddr_state:", loan_df['addr_state'].unique())
print("\ninitial_list_status:", loan_df['initial_list_status'].unique())
print("\napplication_type:", loan_df['application_type'].unique())


# In[17]:


def univariate_analysis(df, columns):
    for col in columns:
        plt.figure(figsize=(10, 5))
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
        plt.show()


# In[18]:


analysis_columns = ['installment']

univariate_analysis(loan_df, analysis_columns)


# In[20]:


plt.figure(figsize=(15, 18))

plt.subplot(3, 2, 1)
sns.barplot(x='grade', y='installment', data=loan_df, palette='viridis')
plt.xlabel('Grade')
plt.ylabel('Installment')
plt.title('Distribution of Grade')

plt.subplot(3, 2, 2)
sns.barplot(x='emp_length', y='installment', data=loan_df, palette='viridis')
plt.xlabel('Employer Length')
plt.ylabel('Installment')
plt.title('Distribution of Employer Length')

plt.subplot(3, 2, 3)
sns.barplot(x='home_ownership', y='installment', data=loan_df, palette='viridis')
plt.xlabel('Home Ownership')
plt.ylabel('Installment')
plt.title('Distribution of Home Ownership')

plt.subplot(3, 2, 4)
sns.barplot(x='verification_status', y='installment', data=loan_df, palette='viridis')
plt.xlabel('Verification Status')
plt.ylabel('Installment')
plt.title('Distribution of Verification Status')

plt.subplot(3, 2, 5)
sns.barplot(x='loan_status', y='installment', data=loan_df, palette='viridis')
plt.xlabel('Loan Status')
plt.ylabel('Installment')
plt.title('Distribution of Loan Status')

plt.tight_layout()
plt.show()


# In[21]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[ ]:




