# pands-project

## My Initial Findings

The Iris flower data set or Fisher's Iris data (also called Anderson's Iris data set) set is a multivariate data set introduced by the British statistician and biologist Ronald Fisher in his 1936 paper "The use of multiple measurements in taxonomic problems". The data set consists of 3 species of Iris plants with 150 samples, 50 from each. The species are Setosa, Versicolor and Virginica. The data contains the sepal length and width, and the petal length and width. So far I have used numpy and pandas to extrapolate some initial data to the console. The next goal is to put this data in a file and some further research. I used pandas.DataFrame.describe to get some extra data such as mean, standard deviation, max and min etc.

References:

https://www.w3schools.com/python/numpy_intro.asp
https://www.w3schools.com/python/pandas/default.asp
https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html
https://www.ntirawen.com/2018/09/iris-dataset-prediction-in-machine.html
http://seaborn.pydata.org/generated/seaborn.FacetGrid.html
https://sklearn.org/

## Analysis

I have created 2 python files. the first one goes through the library sklear which is a library called Scikitlearn.
I take a look at various features of the dataset using numpy and pandas. I then creat 5 histograms. The second file
performs a matrix operation called a transformation on the numpy array of the features.

## Histograms

I added 5 histograms to the project using Matloplib.  Sepal length, sepal width, petal lenght, petal width and one that is just the targets (species). The targets have a one to one relation because there are 50 samples for each species.

