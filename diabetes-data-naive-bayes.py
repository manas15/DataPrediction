# data loading
import numpy as np
import urllib
# url with dataset
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
# download the file
raw_data = urllib.urlopen(url)
# load the CSV file as a numpy matrix
dataset = np.loadtxt(raw_data, delimiter=",")
# separate the data from the target attributes
X = dataset[:,0:8]
y = dataset[:,8]



# data normalization 
from sklearn import preprocessing
# normalize the data attributes
normalized_X = preprocessing.normalize(X)
# standardize the data attributes
standardized_X = preprocessing.scale(X)



# feature selection 
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X, y)
# display the relative importance of each attribute
print("Feature selection")
print(model.feature_importances_)



# naive bayes implementation 
from matplotlib import pyplot
# create model
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()



# fit the naive-bayes model 
model.fit(X, y)
print("Naive-Bayes Model:")
print(model)



# make predictions
expected = y
predicted = model.predict(X)



# summarize the fit of the model
print("classification_report::")
print(metrics.classification_report(expected, predicted))

print("confusion_matrix::")
print(metrics.confusion_matrix(expected, predicted))


# check how many of predictions match actual values 
same = 0
diff = 0
for i in range(len(expected)):
	if expected[i] == predicted[i]:
		same += 1
	else:
		diff += 1
print("same::", same)
print("different::", diff)




# plot the model 
pyplot.figure()
pyplot.plot(X[:,1], expected, 'r.')
pyplot.plot(X[:,1], predicted, 'b.')
pyplot.xlabel("x_1")
pyplot.ylabel("y_1")
pyplot.title("Diabetes data prediction using naive-bayes")
pyplot.show() 
