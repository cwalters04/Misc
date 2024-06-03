#%%
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# importing boston dataset
# df=pd.read_excel('Boston_Housing.xls')
training_set = pd.read_csv('./testdata/train.csv')
testing_set = pd.read_csv('./testdata/test.csv')

# Testing and Training Data Sets
print()
print("Training Set: ")
print(training_set.head())
print(training_set.isnull().sum())
print(training_set.describe())

print()
print("Testing Set:")
print(testing_set.head())
print(testing_set.isnull().sum())
print(testing_set.describe())

# Show pair-plot objects for training_set
print()
print(sns.pairplot(training_set, height=1))

# Show the pairplot points on a graph
col_study = ['zn', 'indus', 'nox', 'rm']
sns.pairplot(training_set[col_study], height=2.5)
plt.show
print(training_set.corr())
# %%
