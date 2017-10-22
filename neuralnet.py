from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

fig, subaxes = plt.subplots(3, 1, figsize=(6, 18))
for units, axes in zip([1, 10, 100], subaxes):
    nnclf = MLPClassifier(hidden_layer_sizes=[units], solver='lbfgs', random_state=0).fit(X_train, y_train)
    title = 'a title'.format(units)
    plt.class_regione_for_classifier_subplot(nnclf, X_train, y_train, X_test, y_test, title, axes)
    plt.tight_layout
