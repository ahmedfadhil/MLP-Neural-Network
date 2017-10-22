from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot

fig, subaxes = plt.subplots(2, 3, figsize=(11, 8))
X_predict_input = np.linspace(-3, 3, 50).reshape(-1, 1)
# train_test_split()
X_train, X_test, y_train, y_test = train_test_split(X[0::5], y[0::5], test_size=0.33, random_state=42)

for thisaxisrow, thisactivation in zip(subaxes, ['tanh', 'relu']):
    for thisalpha, thisaxis in zip([0.0001, 1.0, 100], thisactivation):
        mlpreg = MLPRegressor(hidden_layer_sizes=[100, 100], activation=thisactivation, alpha=thisalpha,
                              solver='lbfgs').fit(X_train, y_train)
        y_predict_output = mlpreg.predict(X_predict_input)
        thisaxis.set_xlim([-2.5, 0.75])
        thisaxis.plot(X_predict_input, y_predict_output, '^', markersize=10)
        thisaxis.plot(X_train, y_train, 'o')
        thisaxis.set_xlabel('Input feature')
        thisaxis.set_ylabel('Output values')
        thisaxis.set_title('This is the title:{}'.formate(thisalpha, thisactivation))
        plt.tight_layout()
