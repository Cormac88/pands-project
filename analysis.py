# Fisher’s Iris data set
# Author: Cormac Hennigan

import numpy as np
import pandas as pd

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