#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, seaborn, sklearn, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from scipy import sparse
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import make_scorer, balanced_accuracy_score, f1_score
from sklearn.metrics import classification_report


# In[8]:


file = Path("ObesityDataSet_raw_and_data_sinthetic.csv")

df = pd.read_csv(file)


df = df.dropna()
df = df.drop(columns=["Height"])
df = df.drop(columns=["Weight"])

df = df.rename(columns={"NObeyesdad":"Obesity_levels"})
df.head()


# In[9]:


X = df.drop(columns=["Obesity_levels"])
y = df["Obesity_levels"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

cat_cols = [
    "Gender",
    "family_history_with_overweight",
    "FAVC",
    "CAEC",
    "SMOKE",
    "SCC",
    "CALC",
    "MTRANS"
]

num_cols = [
    "Age",
    "FCVC",
    "NCP",
    "CH2O",
    "FAF",
    "TUE"
]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
        ("num", StandardScaler(with_mean=False), num_cols),
    ],
    remainder="drop",
    sparse_threshold=0.3, 
)


# In[13]:


#baseline model
from sklearn.dummy import DummyClassifier

#majority vote strategy
clf = LogisticRegression( solver="lbfgs", max_iter = 5000, class_weight = "balanced")

dummy = Pipeline (steps=[ ("preprocess", preprocess), ("dummy", DummyClassifier(strategy="most_frequent", random_state=42))]
)



# In[14]:


clf = LogisticRegression( solver="lbfgs", max_iter = 5000, class_weight = "balanced")

pipe = Pipeline (steps=[ ("preprocess", preprocess), ("model", clf)]
)

pipe.fit(X_train, y_train)


# In[19]:


y_pred = pipe.predict(X_test)


scoring = {
    'f1_macro': 'f1_macro',
    'balanced_accuracy': make_scorer(balanced_accuracy_score)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)

dummyresults=cross_validate(dummy, X, y, cv=cv, scoring = scoring, n_jobs=1)


print(f"Dummy(most_frequent)  F1-macro: {dummyresults['test_f1_macro'].mean():.3f} ± {dummyresults['test_f1_macro'].std():.3f} | "
      f"BalAcc: {dummyresults['test_balanced_accuracy'].mean():.3f} ± {dummyresults['test_balanced_accuracy'].std():.3f}")
print("##################")
print(f"Model F1-macro: {cv_results['test_f1_macro'].mean():.3f} ± {cv_results['test_f1_macro'].std():.3f}")
print(f"Model Balanced Accuracy: {cv_results['test_balanced_accuracy'].mean():.3f} ± {cv_results['test_balanced_accuracy'].std():.3f}")

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print(classification_report(y_test, y_pred, digits=3))



# In[69]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




