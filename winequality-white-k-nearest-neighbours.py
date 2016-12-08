# data loading
import numpy as np
import urllib
dataset = np.loadtxt(open("winequality-white.csv","rb"),delimiter=";",skiprows=1)
# separate the data from the target attributes
X = dataset[:,0:11]
y = dataset[:,11]

# data normalization
from sklearn import preprocessing
# normalize the data attributes
normalized_X = preprocessing.normalize(X)
# standardize the data attributes
standardized_X = preprocessing.scale(X)


#
# feature selection
#
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X, y)

# display the relative importance of each attribute
print(model.feature_importances_)

# naive bayes implementation
from matplotlib import pyplot

# create model
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

# fit a k-nearest neighbor model to the data
model = KNeighborsClassifier()
model.fit(X, y)

print("kNeighbors Model:")
print(model)

# make predictions
expected = y
predicted = model.predict(X)

# summarize the fit of the model
print("classification_report::")
print(metrics.classification_report(expected, predicted))

print("confusion_matrix::")
print(metrics.confusion_matrix(expected, predicted))


# check how far the predictions are from actual values
difference = list()
total_diff = 0.
for i in range(len(y)):
        diff = predicted[i] - expected[i]
        difference.append(diff)
        total_diff += abs(diff)
diversion = total_diff / len(y)


# check how many of predictions match actual values
same = 0
diff = 0
for i in range(len(expected)):
        if expected[i] == predicted[i]:
                same += 1
        else:
                diff += 1

print "     Report                            "
print "     Number of rows                    ", len(y)
print "     Number of attributes              ", len(X[0])
print "     Error                             ", diversion
print "     Error percentage                  ", round(diversion * 100, 4)
print "     Number of exact Matches           ", same
print "     Number of times it didn't match   ", diff




# plot the model
pyplot.figure()
pyplot.plot(X[:,1], expected, 'r.')
pyplot.plot(X[:,1], predicted, 'b.')
pyplot.xlabel("x_1")
pyplot.ylabel("y_1")
pyplot.title("WineQuality-red pyearth data prediction")
pyplot.show()


