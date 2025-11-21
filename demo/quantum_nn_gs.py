
import time
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import json
import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import MinMaxScaler

from demo_tools import calc_features, get_data, compute_metrics
from demo_parkinson_nn import encode_input_as_circuit, create_model

class QuantumNNRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_neurons=100, learning_rate=0.01, activation='sigmoid', epochs=100):
        self.n_neurons = n_neurons
        self.learning_rate = learning_rate
        self.activation = activation
        self.epochs = epochs
        self.qubits = [cirq.GridQubit(0, i) for i in range(2)]

    def fit(self, X, y):
        self.model = self._build_model(input_dim=X.shape[1])

        self.datapoints = tfq.convert_to_tensor([
            encode_input_as_circuit(x, self.qubits) for x in X
        ])
        self.commands = tf.convert_to_tensor(X.astype(np.float32))
        self.y_scaler = MinMaxScaler(feature_range=(-1, 1))
        y_scaled = self.y_scaler.fit_transform(y)

        self.model.fit([self.datapoints, self.commands], y_scaled, epochs=self.epochs, verbose=0)
        return self

    def predict(self, X):
        datapoints_test = tfq.convert_to_tensor([
            encode_input_as_circuit(x, self.qubits) for x in X
        ])
        commands_test = tf.convert_to_tensor(X.astype(np.float32))
        preds_scaled = self.model([datapoints_test, commands_test]).numpy()
        return self.y_scaler.inverse_transform(preds_scaled)

    def _build_model(self, input_dim):
        controller = tf.keras.Sequential([
            tf.keras.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(self.n_neurons, activation=self.activation),
            tf.keras.layers.Dense(6)
        ])
        model, _, _ = create_model(controller, self.qubits, input_dim)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model


def run_quantum_gridsearch(X_train, Y_train, X_test, Y_test, times, CV_times, r2, MSE):
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler

    start_time = time.time()

    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)

    param_grid = {
        'n_neurons': [50, 75, 100],
        'learning_rate': [0.01, 0.005, 0.001],
        'activation': ['sigmoid', 'relu', 'tanh'],
        'epochs': [50]
    }

    grid = GridSearchCV(
        estimator=QuantumNNRegressor(),
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        verbose=2,
        n_jobs=1
    )
    grid.fit(X_train_scaled, Y_train)

    best_model = grid.best_estimator_
    print("Best HParams:", grid.best_params_)
    with open('quantum_nn_hparams.json', 'w') as f:
        json.dump(grid.best_params_, f, indent=4)
    datapoints_test = tfq.convert_to_tensor([
        encode_input_as_circuit(x, best_model.qubits) for x in X_test_scaled
    ])
    commands_test = tf.convert_to_tensor(X_test_scaled.astype(np.float32))

    result = compute_metrics(
        model=best_model.model,
        x_tr=[best_model.datapoints, best_model.commands],
        y_tr=Y_train,
        x_ts=[datapoints_test, commands_test],
        y_ts=Y_test,
        model_type='quantum',
        y_scaler=best_model.y_scaler
    )

    train_predict, test_predict, train_accuracy, test_accuracy, train_MSE, test_MSE = result
    times.append(time.time() - start_time)
    CV_times.append(grid.cv)
    r2.append(test_accuracy)
    MSE.append(test_MSE)

if __name__=="__main__":
    times = []
    CV_times = []
    r2 = []
    MSE = []
    X, Y, dataset = get_data()
    X_train, Y_train, X_test, Y_test = calc_features(X, Y, dataset)
    run_quantum_gridsearch(X_train, Y_train, X_test, Y_test, times, CV_times, r2, MSE)
