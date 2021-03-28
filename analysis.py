# Fisher’s Iris data set
# Author: Cormac Hennigan

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

columns = ['sepal_length','sepal_width','petal_length','petal_width','type']
iris = pd.read_csv("Iris Data.csv", names = columns)

print("First five rows")
print(iris.head())
print("*********")
print("columns",iris.columns)
print("*********")
print("shape:",iris.shape)
print("*********")
print("Size:",iris.size)
print("*********")
print("no of samples available for each type") 
print(iris["type"].value_counts())
print("*********")
print(iris.describe())

iris_setosa = iris.loc[iris["type"] == "Iris-setosa"]
iris_virginica = iris.loc[iris["type"] == "Iris-virginica"]
iris_versicolor = iris.loc[iris["type"] == "Iris-versicolor"]

sns.FacetGrid(iris,hue="type",size=3).map(sns.distplot,"petal_length").add_legend()
plt.savefig("Petal Length")
sns.FacetGrid(iris,hue="type",size=3).map(sns.distplot,"petal_width").add_legend()
plt.savefig("Petal Width")
sns.FacetGrid(iris,hue="type",size=3).map(sns.distplot,"sepal_length").add_legend()
plt.savefig("Sepal Length")
sns.FacetGrid(iris,hue="type",size=3).map(sns.distplot,"sepal_width").add_legend()
plt.savefig("Sepal Width")