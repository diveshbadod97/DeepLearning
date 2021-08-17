import numpy as np
from sklearn import linear_model, datasets
from sklearn.metrics import mean_squared_error
#from math import floor, ceil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.svm import SVC


def example1():
    #Load Train and Test datasets
    diabetes = datasets.load_diabetes()
    X, y = diabetes.data, diabetes.target
    y = y[:, np.newaxis] # create matrix versions of this array
    X = X[:,np.newaxis,2] #extract third column from X

    #Identify feature and response variable(s) and values must be numeric and numpy arrays
    n_samples = len(diabetes.target)  #should be 442
    x_train = X[:int(n_samples / 2)]  #first half of samples (221 samples)
    y_train = y[:int(n_samples / 2)]  
    x_test = X[int(n_samples / 2):]  #second half of samples (221 samples)
    y_test = y[int(n_samples / 2):]  

    # Create linear regression object
    linear = linear_model.LinearRegression()
    # Train the model using the training sets and check score
    linear.fit(x_train, y_train)

    #Equation coefficient and Intercept
    print('Coefficient: \n', linear.coef_)
    print('Intercept: \n', linear.intercept_)
    #Predict Output
    predicted= linear.predict(x_test)
    #mean squared error
    mse1 = sum((predicted-y_test)*(predicted-y_test))/len(y_test)
    mse2 = np.mean((predicted-y_test) ** 2)
    mse = mean_squared_error(y_test, predicted)  #or use built-in function
    print("Mean squared error: %.2f" % mse )
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % linear.score(x_train, y_train))

    #plot output
    plt.plot(x_train, y_train,'ro')
    plt.plot(x_test , linear.predict(x_test), color='blue',linewidth=3)
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.title('Linear Fit')
    plt.savefig('example1.png')


def example2_classifier():
    # create dataset
    X, y = make_moons(noise=0.3, random_state=0)
    X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=.4, random_state=42)

    # mesh data for plots
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))

    # plot the dataset 
    plt.figure(1)
    plt.title("Input data")
    # Plot the training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
    # and testing points
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, alpha=0.6)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    #plt.xticks(())  #use to remove ticks
    #plt.yticks(())


    # build classifier
    name = "Linear SVM"
    classifier= SVC(kernel="linear", C=0.025)
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)

    # plot mesh of classifier
    plt.figure(2)
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    Z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)

    # Plot also the training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    # and testing points
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
            alpha=0.6)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
        
    plt.title(name)
    plt.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
            size=15, horizontalalignment='right')
    plt.savefig('example2_classifier.png')