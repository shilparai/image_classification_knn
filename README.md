# image_classification_knn
<img height = "204" src = "https://github.com/shilparai/image_classification_knn/blob/master/knn_image.PNG">
*Determine the class of these two red points?

# Introduction

k-Nearest Neighbour is the most simple machine learning and image classification algorithm. This algorithm depends on the distance between features vectors. In our cases, these features are pixel values in image matrix (height x width)
k-NN algorithm classifies new unknown data points by finding the most common class among the k-closet examples.

k-NN can also be used for regression. In this case, output is a continuous variable which is the average of k-closet data points.

k-NN is a non-parametric learning algorithm, which means that it doesn't assume anything about the underlying data

Some useful links

[A Quick Introduction to K-Nearest Neighbors Algorithm](https://medium.com/@adi.bronshtein/a-quick-introduction-to-k-nearest-neighbors-algorithm-62214cea29c7)

[K-Nearest Neighbors Algorithm in Python and Scikit-Learn](https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/)

# Pros

1. One main advantage of k-NN algorithm is that it's simple to implement and understand.
2. It is lazy learning algorithm and therefore requires no training prior to making real time predictions. This makes the KNN algorithm much faster than other algorithms that require training e.g SVM, linear regression, etc.
3. There are only two parameters required to implement KNN i.e. the value of K and the distance function (e.g. Euclidean or Manhattan etc.)

# Cons
1. k-NN is more suited for low-dimensional features spaces(which images are not). Distances in high-dimensional features spaces are often unintuitive (curse of dimensionality)
2. k-NN doesn't learn anything, that means it does not work towrads improving the performance(error) by looking at previous steps
3. Sharing of k-NN models can be a problem when the data size is really huge as it requires to share all the data.
4. k-NN algorithm doesn't work well with categorical features since it is difficult to find the distance between dimensions with categorical features.

[DATASET](https://www.kaggle.com/c/dogs-vs-cats/data)

# Prerequisite
1. Python3
2. OpenCV 3
3. Packages mentioned in knn.py 
4. Pycharm/ Jupyter Notebook

# How to use
1. Download the dataset and save it in the folder "datasets"
2. Name of sub-folders in parent folder "datasets" represnt lables (class). In our case, it's "Dog", and "Cat"
3. Clone the repository
4. Run knn.py using cmd/pycharm/jupyter notebook

Happy Learning !!!
