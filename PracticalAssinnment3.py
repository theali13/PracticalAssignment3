import numpy as np
import pandas as pd
from matplotlib import pyplot
from termcolor import colored


def featureNormalize(X):
    X_norm = X.copy()
    mu = np.zeros(X_norm.shape[1])
    sigma = np.zeros(X_norm.shape[1])
    for c in range(X.shape[1]):
        mu[c] = np.mean(X_norm[:, c])
    for v in range(X.shape[1]):
        sigma[v] = np.std(X_norm[:, v])
    print(mu)
    print(sigma)
    for k in range(X_norm.shape[0]):
        # print(i)
        for z in range(X_norm.shape[1]):
            X_norm[k][z] = X_norm[k][z] - mu[z]
            X_norm[k][z] = X_norm[k][z] / sigma[z]

    return X_norm, mu, sigma


def kfold(order, dataFrame, arraynodate, training_rows, cross_validation_rows, test_rows):
    numpydataall = dataFrame.values
    prices = numpydataall[:, 2]
    numpydataall = numpydataall[:, arraynodate]

    X_norm, mu, sigma = featureNormalize(numpydataall)
    # print(X_norm)

    # divide the normalized data into train validate and test
    X_norm = np.concatenate([np.ones((prices.size, 1)), X_norm], axis=1)

    if (order == '0'):
        numpydatacross = X_norm[0:int(cross_validation_rows - 1)]
        numpydatatrain = X_norm[int(cross_validation_rows):int(cross_validation_rows + training_rows - 1)]
        numpydatatest = X_norm[int(training_rows + cross_validation_rows):int(
            training_rows + cross_validation_rows + test_rows)]
        pricescross = prices[0:int(cross_validation_rows - 1)]
        pricestrain = prices[int(cross_validation_rows):int(cross_validation_rows + training_rows - 1)]
        pricestest = prices[int(training_rows + cross_validation_rows):int(
            training_rows + cross_validation_rows + test_rows)]

    if (order == '1'):
        numpydatatest = X_norm[0:int(test_rows - 1)]
        numpydatacross = X_norm[int(test_rows):int(cross_validation_rows + test_rows - 1)]
        numpydatatrain = X_norm[int(test_rows + cross_validation_rows):int(training_rows + cross_validation_rows + test_rows)]
        pricestest = prices[0:int(test_rows - 1)]
        pricescross = prices[int(test_rows):int(cross_validation_rows + test_rows - 1)]
        pricestrain = prices[int(test_rows + cross_validation_rows):int(training_rows + cross_validation_rows + test_rows)]


    return numpydatatrain, numpydatacross, numpydatatest,pricestrain,pricescross,pricestest


def computeCosttest(X, y):
    m = y.size


    # h = np.dot(X, theta)
    J = sum((np.subtract(X, y) ** 2)) * (1 / (2 * m))


    return J


def computeCostMulti(X, y, theta, lam):
    m = y.shape[0]

    h = np.dot(X, theta)
    J = (1 / (2 * m)) * (np.dot((np.subtract(h, y)).T, np.subtract(h, y))) + (
                (lam / (2 * m)) * (np.sum(np.square(theta))))

    return J


def computeCostMultiCV(X, y, theta):
    m = y.shape[0]

    h = np.dot(X, theta)
    J = (1 / (2 * m)) * (np.dot((np.subtract(h, y)).T, np.subtract(h, y)))

    return J


def gradientDescentMulti(X, y, theta, alpha, lam, iters):
    m = y.shape[0]
    theta = theta.copy()
    lam = lam

    J_history = []

    for i in range(iters):
       
        hypothesis = np.dot(X, theta)
        alphabym = alpha / m
        theta = (theta * (1 - ((alpha * lam) / m))) - ((alpha / m) * (np.dot(X.T, hypothesis - y)))

        J_history.append(computeCostMulti(X, y, theta, lam))

    return theta, J_history


arraynodate = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
dataFrame = pd.read_csv('trainingdata.csv')
my_data = np.genfromtxt('trainingdata.csv', delimiter=',')
column_names = ['id', 'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view',
                'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat',
                'long', 'sqft_living', 'sqft_lot15']
training_rows = np.ceil(0.6 * 17998)
cross_validation_rows = np.ceil(0.2 * 17998)
test_rows = np.floor(0.2 * 17998)
dataFrame = dataFrame.loc[0:17998]
training_data = dataFrame.loc[0:training_rows - 1]
cross_validation_data = dataFrame.loc[training_rows: training_rows + cross_validation_rows - 1]
test_data = dataFrame.loc[training_rows + cross_validation_rows: training_rows + cross_validation_rows + test_rows]
numpydataall = np.zeros(dataFrame.shape)
print(dataFrame.iloc[0][column_names[0]])
print(training_data.shape[1])
# change data frame to numpy using loop/ method
# for i in range(dataFrame.shape[0]):
#    for j in range(dataFrame.shape[1]-1):
#        numpydataall[i][j] = float(dataFrame.iloc[i][column_names[j]])
#        print(i)
#        print(j)

numpydataall = dataFrame.values
prices = numpydataall[:, 2]
numpydataall = numpydataall[:, arraynodate]

# print(numpydataall)
# numpydatatrain = numpydataall[0:int(training_rows-1)]
# numpydatatrainsq = numpydatatrain**2
# numpydatacross = numpydataall[int(training_rows):int(training_rows+cross_validation_rows-1)]
# numpydatacrosssq = numpydatacross**2
# numpydatatest = numpydataall[int(training_rows+cross_validation_rows):int(training_rows+cross_validation_rows+test_rows)]
# numpydatatestsq = numpydatatest**2
# print(numpydatatrain)
# print('second')
# print(numpydatacross)
# print('third')
# print(numpydatatest)
# print(training_data.iloc[0][column_names[1]])
# Normalize the data
# print(featureNormalize(numpydataall[:,:-1]))
X_norm, mu, sigma = featureNormalize(numpydataall)
# print(X_norm)

# divide the normalized data into train validate and test
X_norm = np.concatenate([np.ones((prices.size, 1)), X_norm], axis=1)
numpydatatrain = X_norm[0:int(training_rows - 1)]
# numpydatatrainsq = np.power(numpydatatrain,0.5)
numpydatacross = X_norm[int(training_rows):int(training_rows + cross_validation_rows - 1)]
# numpydatacrosssq = np.power(numpydatacross,0.5)
numpydatatest = X_norm[
                int(training_rows + cross_validation_rows):int(training_rows + cross_validation_rows + test_rows)]
# numpydatatestsq = np.power(numpydatacross,0.5)
# print(numpydatatrainsq)
# print('square')
# print(numpydatatrain)
# print(numpydatacrosssq)
# print(numpydatatrainsq)
# construct hypothesis
pricestrain = prices[0:int(training_rows - 1)]
pricescross = prices[int(training_rows):int(training_rows + cross_validation_rows - 1)]
pricestest = prices[int(training_rows + cross_validation_rows):int(
                  training_rows + cross_validation_rows + test_rows)]
# theta = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
for fold in range(3):
    alpha = 0.35
    num_iters = 50
    lam = [0.01, 1, 10, 100]
    # init theta and run gradient descent
    theta = np.zeros(20)
    for x in range(len(lam)):
        theta, J_history = gradientDescentMulti(numpydatatrain, pricestrain, theta.copy(), alpha,lam[x], num_iters)

        # Plot the convergence graph
        pyplot.plot(np.arange(len(J_history)), J_history, lw=2)
        pyplot.xlabel('Number of iterations')
        pyplot.ylabel('Cost J')
        pyplot.legend(['hypo1'])
        pyplot.show()
        # Display the gradient descent's result
        print(J_history)
        print('theta computed from gradient descent: {:s}'.format(str(theta)))
        costcv = computeCostMultiCV(numpydatacross,
                                    pricescross, theta)
        print(costcv)
        predicted = np.dot(numpydatatest, theta)
        print(colored('error test', 'red'),
              computeCosttest(predicted, pricestest))
        # print(prices)
        print(colored('Estimated', 'red'))
        print(predicted)
        print(colored('Real', 'red'))
        print(pricestest)
        #pricestest = prices[
         #            int(training_rows + cross_validation_rows):int(training_rows + cross_validation_rows + test_rows)]
        print(colored('Accuracy', 'red'))
        accuracy = np.zeros(predicted.shape[0])
        for b in range(predicted.shape[0]):
            if (predicted[b] < pricestest[b]):
                accuracy[b] = predicted[b] / pricestest[b]
            else:
                accuracy[b] = pricestest[b] / predicted[b]

        print(accuracy)
        print(np.mean(accuracy) * 100, '%', 'lamda =', lam[x])

    #print(order)
    #print(numpydatatrain.shape)


    #print(np.mean(np.divide(predicted ,prices[int(training_rows+cross_validation_rows):int(training_rows+cross_validation_rows+test_rows)])))
    # Estimate the price of a 1650 sq-ft, 3 br house
    # ======================= YOUR CODE HERE ===========================
    # Recall that the first column of X is all-ones.
    # Thus, it does not need to be normalized.
    #value = np.array([1650,3])
    #normalize_test_data = np.subtract(X[:,0:3],mu[1])
    #normalize_test_data = None
    #another =
    #price = 0
    # print 'Predicted price of a 1650 sq-ft, 3 br house:', price
    #normalize1 = (value - mu) / sigma
    #normalize2 =np.append(1,normalize1)
    #price = np.dot(normalize2,theta)
    # ===================================================================
    #print(price)
    #print(normalize2)
    #print(np.stack([np.ones(1),(value - mu) / sigma] ,axis = 1)
    #print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): ${:.0f}'.format(price))
    alpha = 0.35
    num_iters = 50
    lam = [0.01,1,10,100]
    hypo2 = [1,2,3,5,6,7,9,10,11,13,15,17]
    # init theta and run gradient descent
    #print(numpydatatrain[:,hypo2])
    theta = np.zeros(12)
    for z in range(len(lam)):
        theta, J_history = gradientDescentMulti(numpydatatrain[:,hypo2], pricestrain, theta, alpha,lam[z], num_iters)

        # Plot the convergence graph
        pyplot.plot(np.arange(len(J_history)), J_history,mec = 'k', lw=2)
        pyplot.xlabel('Number of iterations')
        pyplot.ylabel('Cost J')
        pyplot.legend(['hypo2']);
        pyplot.show()
        # Display the gradient descent's result
        print(J_history)
        print('theta computed from gradient descent: {:s}'.format(str(theta)))
        costcv = computeCostMultiCV(numpydatacross[:,hypo2],pricescross,theta)
        print(costcv)
        predicted = np.dot(numpydatatest[:,hypo2],theta)
        print(colored('error test','red'), computeCosttest(predicted, pricestest))
        print(colored('Estimated', 'red'))
        print(predicted)
        print(colored('Real', 'red'))
        print(pricestest)
        #pricestest = prices[int(training_rows+cross_validation_rows):int(training_rows+cross_validation_rows+test_rows)]
        print(colored('Accuracy 2', 'red'))
        accuracy = np.zeros(predicted.shape[0])
        predicted = np.multiply(predicted,-1)
        for b in range(predicted.shape[0]):
            if(predicted[b] < pricestest[b]):
                accuracy[b] = predicted[b]/pricestest[b]
            else:
                accuracy[b] = pricestest[b]/predicted[b]

        print(accuracy)
        print(np.mean(accuracy) * 100,'%')


    alpha = 0.3
    num_iters = 50
    lam = [0.01,1,10,100]
    hypo3 = [0,1,4,5,6,7,8,11,12,13,14,15,16,18]
    # init theta and run gradient descent
    #print(numpydatatrain[:,hypo2])
    theta = np.zeros(14)
    for c in range(len(lam)):
        theta, J_history = gradientDescentMulti(numpydatatrain[:,hypo3], pricestrain, theta, alpha,lam[c], num_iters)
        # Plot the convergence graph
        pyplot.plot(np.arange(len(J_history)), J_history,mec = 'k', lw=2)
        pyplot.xlabel('Number of iterations')
        pyplot.ylabel('Cost J')
        pyplot.legend(['hypo3']);
        pyplot.show()
        # Display the gradient descent's result
        print(J_history)
        print('theta computed from gradient descent: {:s}'.format(str(theta)))
        costcv = computeCostMultiCV(numpydatacross[:,hypo3],pricescross,theta)
        print(costcv)
        predicted = np.dot(numpydatatest[:,hypo3],theta)
        print(colored('error test', 'red'),
              computeCosttest(predicted, pricestest))
        #print(prices)
        print(colored('Estimated', 'red'))
        print(predicted)
        print(colored('Real', 'red'))
        print(pricestest)
        #pricestest = prices[int(training_rows+cross_validation_rows):int(training_rows+cross_validation_rows+test_rows)]
        print(colored('Accuracy 3', 'red'))
        accuracy = np.zeros(predicted.shape[0])
        for b in range(predicted.shape[0]):
            if(predicted[b] < pricestest[b]):
                accuracy[b] = predicted[b]/pricestest[b]
            else:
                accuracy[b] = pricestest[b]/predicted[b]

        print(accuracy)
        print(np.mean(accuracy) * 100,'%')
    order = str(fold)
    if (fold < 2):
            numpydatatrain, numpydatacross, numpydatatest, pricestrain, pricescross, pricestest = kfold(order,dataFrame, arraynodate, training_rows,cross_validation_rows,test_rows)
    print(fold)