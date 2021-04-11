# pands-project

## Introduction

The Iris flower data set or Fisher's Iris data (also called Anderson's Iris data set) set is a multivariate data set introduced by the British statistician and biologist Ronald Fisher in his 1936 paper "The use of multiple measurements in taxonomic problems". The data set consists of 3 species of Iris plants (Setosa, Versicolor and Virginica). Edgar Anderson collected 50 samples of the flower, of which each was a different species. He collected 150 in total. For each sample he measured the sepal length and width, and the petal length and width along with the corresponding species. The data contains the sepal length and width, and the petal length and width. Picture of an Iris which is a flower. 

## Training a machine model with scikit-learn

This is what the data looks like in comma separated variables (CSV) format. For example, this first row indicates that Anderson found... This particular Iris was the species Setosa. The dataset also contains Iris measurements from the species Versicolor, and the species Virginica. In 1936, Sir Ronald Fisher wrote a paper about the Iris dataset, specifically about how a technique called **Linear Discriminant Analysis** could be used to accurately distinguish the three species from one another using only the sepal and petal measurements. In other words, Fisher framed this as a supervised learning problem in which we are attempting to predict the species of the given Iris using the available data. This is supervised learning, because we are trying to learn the relationship between the data namely the iris measurements and the outcome, which is the species of Iris. If this was unlabelled data, meaning that we only had the measurements and not the species, we might frame this as unsupervised learning by attempting to cluster the samples into meaningful groups. The Iris dataset has become a famous dataset for machine learning, because it turns out to be an easy supervised learning task. There is a strong relationship between the measurements and the species, and thus various machine learning models can accurately predict the species given the measurements. The dataset is described in more depth in the UCI Machine Learning Repository, which is a repository of hundreds of datasets that are suitable for machine learning tasks. Because the Iris dataset is so popular as a **toy dataset**, it has been built into the **scikit-learn library.** In this project I load the Iris data from scikit-learn and examine it so that I can use a machine learning model to predict species using the iris measurements.

I started by writing:

`from sklearn.datasets import load_iris`

This imports the load_iris function from the sklearn.datasets module. This is because the convention in scikit-learn is to import individual modules, classes or functions instead of importing scikit-learn as a whole.  I then ran the load_iris function and saved the return value in an object called iris. This object is a special container called a "bunch", which is sci-kit learn's special object type for storing datasets and their attributes. One of those attributes is called "data". This is the same data we saw previously from the csv file with four rows and 150 columns. Each row represents one flower, and the four columns represent the four measurements (petal length, petal width, sepal length and sepal width). I'm now going to introduce some important machine learning terminology that I'll be using throughout this project. Each row is know as an observation. Some equivalent terms are sample, example, instance and record. We can say that the Iris dataset has 150 observations. Each column is know as a feature. Some equivalent terms are predictor, attribute, independent variable, input, regressor and covariated. We can say that the Iris dataset has 4 features. In my analysis program, I printed out an attribute of the iris object called feature_names. This represents the names of the four features. They can also be imagined as the column headers for the data. I also printed out two more attributes called `target` and `target_names`. The target represents what I am going to predict. A zero represents Setosa, a one represents Versicolor and a two represents Virginica. Some equivalent terms for target are **response**, outcome, outcome, label, dependent variable. I will use the term response throughout this project.

The last pieces of terminology that will be used are the two types of supervised learning, which are **classification** and **regression**. A classification problem is one in which the response being predicted is categorical, meaning that its values are in a **finite**, **unordered set**. Predicting the species of Iris is an example of a classification problem, as is predicting if an email is spam or not. In contrast, a regression problem is one in which the response being predicted is **ordered** and **continuous**, such as the price of a house or the height of a person. When looking at `iris.target`, we might wonder how we can tell that this is a classification problem and not a regression problem, since all we can see are the numbers 0, 1 and 2. The answer is that we cannot tell the difference. As we explore the problem, we have to understand how the data is encoded and decide whether the response variable is suited for **classification** or **regression**. In this case we know that the numbers 0, 1 and 2 represent unordered categories, and thus we know to use classification techniques and not regression techniques in order to solve this problem.

The first step in machine learning is for the model to learn the relationship between the **features** and the **response**. Firstly, we need to make sure that the features and response are in the form that scikit-learn expects. There are four key requirements to keep in mind which are as follows:

1. Scikit-learn expects the **features** and the **response** to be passed into the machine learning model as separate objects. `iris.data` and `iris.target` satisfy this condition since they are stored separately.
2. Scikit-learn only expects to see numbers in the **features** and **response** objects. This is why `iris.target` is stored as zero's, one's and two's instead  of the strings setosa, versicolor and virginica. In scikit-learn, the response object should always be numeric regardless of whether it is a regression problem or a classification problem.
3. Scikit-learn expects the **features** and the **response** to be stored as **NumPy** arrays. NumPy is a library for scientific computing that implements a homogenous, multidimensional array knows as an nd array that has been optimised for fast computation. Both `iris.data` and `iris.target` are already stored as nd arrays. 
4. The feature and response objects are expected to have certain shapes. Specifically, the feature obkect should have two dimensions in which the first dimension represented by rows is the number of observations, and the second dimension, represented by columns is the number of features. All NumPy arrays have a shape attribute and so we can verify that the shape of `iris.data` is 150x4. The response object is expected to have a single dimension, and that dimension should have the same magnitude as the first dimension of the feature object. In other words, there should be one response corresponding to each observation. We can verify that the shape of `iris.target` is simply 150. 

I have now verified that `iris.data` and `iris.target` meet scikit-learn's four requirements for feature and response objects. The scikit-learn convention is for the feature data to be stored in an object named 'X', and for the response data to be stored in an object named 'y'. We'll store `iris.data` in 'X' and `iris.target` in 'y'. The 'X' is capitalised because it represents a matrix and the 'y' is lower case because it represents a vector. 



## Analysis

I have created 2 python files. the first one goes through the library sklear which is a library called Scikitlearn.
I take a look at various features of the dataset using numpy and pandas. I then creat 5 histograms. The second file
performs a matrix operation called a transformation on the numpy array of the features.



## Histograms

I added 5 histograms to the project using Matloplib.  Sepal length, sepal width, petal lenght, petal width and one that is just the targets (species). The targets have a one to one relation because there are 50 samples for each species.



## References

https://www.w3schools.com/python/numpy_intro.asp
https://www.w3schools.com/python/pandas/default.asp
https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html
https://www.ntirawen.com/2018/09/iris-dataset-prediction-in-machine.html
http://seaborn.pydata.org/generated/seaborn.FacetGrid.html
https://sklearn.org/

