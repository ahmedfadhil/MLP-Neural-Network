from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

fig, subaxes = plt.subplots(4, 1, figsize=(6, 23))
for this_alpha, axis in zip([0.01, 0.1, 1.0, 5.0], subaxes):
    nnclf = MLPClassifier(solver='lbfgs', activation='tanh',alpha=this_alpha, hidden_layer_sizes=[100,100],random_state=0)
    nnclf.fit(X_train,y_train)
    title = 'this is the title'.format(this_alpha)
    plot_class_regions_for_classifier_subplot(nnclf,X_train,y_train, X_test,y_test,title,axis)
    plt.tight_layout()
