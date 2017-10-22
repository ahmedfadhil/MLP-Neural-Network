from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot

scaler = MinMaxScaler()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_train_scaled= scaler.fit_transform(X_train)
X_test_scaled= scaler.fit_transform(X_test)

clf = MLPClassifier(hidden_layer_sizes=[100,100],solver='lbfgs', random_state=0,alpha=5.0).fit(X_train_scaled,y_train)

print('Breast cancer dataset')
print('Accuracy of NN classifier on training set:{:.2f}'.format(clf.score(X_train_scaled,y_train)))
print('Accuracy of NN classifier on testing data:{.2f}'.format(clf.score(X_test_scaled,y_test)))


