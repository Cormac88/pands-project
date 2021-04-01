# Fisher’s Iris data set
# Author: Cormac Hennigan

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

iris = load_iris()
# print(iris)

# print(iris.data.shape) # shows dimentions. 150 rows, 4 columns or attributes (elements)
# print(iris.target.shape) # One dimention
# print(iris.feature_names)
# print(iris.target_names) # targets are the names of the species of iris. They are a numpy array
# print(iris.DESCR) # Attribut info, summary status etc

# Convert data into Pandas dataframe

df = pd.DataFrame(iris.data, columns = iris.feature_names)
# print(df.head()) # First 5 entries
# print(df.tail())

# Include the target

df['target'] = iris.target # shows the targets (species) as numerical values. 0, 1 and 2.
# print(df.head())

# print(df.dtypes) # Shows the data types. The petals and sepals are floats and the targets are ints.

# print(df.describe())
#print(df.groupby('target').size()) # Shows that there are 50 entries for each species

# Data Visualistaton

df.hist(figsize=(12, 12))
# plt.show()
plt.savefig('Histograms')


# 5 Histograms. Sepal length, sepal width, petal lenght, petal width and one that is just the targets(species).
# The targets have a one to one relation because there are 50 samples for each species.