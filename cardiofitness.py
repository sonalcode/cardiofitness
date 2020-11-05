import numpy as np
import pandas as pd
data= pd.read_csv('C://mpw/CardioGoodFitness.csv')
data.head()

data.describe(include='all')
import matplotlib.pyplot as plt
%matplotlib inline
data.hist(figsize=(10,15))
# box plot between gender and age

import seaborn as sns
sns.boxplot(x="Gender",y="Age",data=data)

# getting product purchased by no. of males and females
pd.crosstab(data['Product'],data['Gender'])

sns.countplot(x='Product',hue="Gender",data=data)
# pairplot is also called lazy plot
sns.pairplot(data)

data['Age'].std()
data['Age'].mean()
sns.distplot(data['Age'])

##bivariate statistics
#distributions of two variables

#cavariance are used in dimension reduction, cavariance= variance(x) = std(x)^2
sns.heatmap(corr, annot=True)
#predicting miles travelled according to usage and fitness
from sklearn import linear_model
regr = linear_model.LinearRegression()
y= data['Miles']
x= data[['Usage','Fitness']]
regr.fit(x,y)
regr.coef_
regr.intercept_

