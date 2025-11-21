
import glob
import time

import pandas as pd
pd.set_option("display.max_columns", None)

import numpy as np
import sympy
import cirq
from cirq.contrib.svg import SVGCircuit
import cairosvg

import matplotlib.pyplot as plt

import tensorflow_quantum as tfq
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from demo_tools import calc_features, get_data, compute_metrics

EPOCHS = 50
N_NEURONS = 75
ACTIVATION = 'relu'
LEARNING_RATE = 0.005

"""
Accuracy on training dataset: 98.00%
Accuracy on test dataset: 97.16%
Mean Squared Error on training samples: 1.1945
Mean Squared Error on test samples: 1.4726
 | Neural Network | GridSearch Cross-Validation. Total fit and predict time: 73.77319812774658 seconds
"""
def run_simple_nn_model(X_train, Y_train, X_test, Y_test, times, CV_times, r2, MSE):
    # Scale X
    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)

    # Scale Y
    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    Y_train_scaled = y_scaler.fit_transform(Y_train)

    # Build and compile model
    # Best HParams: {'activation': 'relu', 'epochs': 50, 'learning_rate': 0.005, 'n_neurons': 75}
    model = Sequential([
        Dense(N_NEURONS, activation=ACTIVATION, input_shape=(X_train.shape[1],)),
        Dense(2)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='mse', metrics=['mse', r2_metric])

    # Train
    start_time = time.time()
    model.fit(X_train_scaled, Y_train_scaled, epochs=EPOCHS, verbose=0)
    result = compute_metrics(model, X_train_scaled, Y_train, X_test_scaled, Y_test, model_type='keras', y_scaler=y_scaler)
    train_predict, test_predict, train_accuracy, test_accuracy, train_MSE, test_MSE = result
    times.append(time.time() - start_time)
    CV_times.append(0)
    print(" | Neural Network | GridSearch Cross-Validation. Total fit and predict time: %s seconds" % (time.time() - start_time))
    r2.append(test_accuracy)
    MSE.append(test_MSE)

def encode_input_as_circuit(x: np.ndarray, qubits: list) -> cirq.Circuit:
    # Create a quantum circuit that encodes input vector x onto qubits using RX, RY, RZ.
    circuit = cirq.Circuit()
    for i, val in enumerate(x[:len(qubits)]):
        circuit.append([
            cirq.rx(val)(qubits[i]),
            cirq.ry(val)(qubits[i]),
            cirq.rz(val)(qubits[i])
        ])
    return circuit

def r2_metric(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


def create_model(controller: tf.keras.Sequential, qubits: list, input_dim: int):
    # Inputs
    circuits_input = tf.keras.Input(shape=(), dtype=tf.string, name='circuits_input')
    commands_input = tf.keras.Input(shape=(input_dim,), dtype=tf.float32, name='commands_input')

    # Classical controller: input → 6 angles (3 per qubit)
    dense_2 = controller(commands_input)
    param_split = tf.split(dense_2, num_or_size_splits=2, axis=1)

    # Symbolic parameter names
    symbols = [f'theta_{i}' for i in range(6)]

    # Per-qubit circuit with 3 params + optional entanglement
    pqc_layers = []
    for i in range(2):
        circuit = cirq.Circuit(
            cirq.rz(sympy.Symbol(symbols[3*i + 0]))(qubits[i]),
            cirq.ry(sympy.Symbol(symbols[3*i + 1]))(qubits[i]),
            cirq.rx(sympy.Symbol(symbols[3*i + 2]))(qubits[i])
        )
        if i == 0:
            circuit.append(cirq.CNOT(qubits[0], qubits[1]))

        pqc = tfq.layers.ControlledPQC(circuit, operators=cirq.Z(qubits[i]))
        out = pqc([circuits_input, param_split[i]])
        pqc_layers.append(out)

    # Draw last circuit
    svg_data = SVGCircuit(circuit)._repr_svg_()
    svg_data = svg_data.replace('font-family="Arial"', 'font-family="DejaVu Sans"')
    with open("../images/circuit-nn.svg", "w", encoding="utf-8") as f:
        f.write(svg_data)
    cairosvg.svg2png(bytestring=svg_data.encode("utf-8"), write_to="../images/circuit-nn.png")

    output = tf.keras.layers.Concatenate()(pqc_layers)

    model = tf.keras.Model(inputs=[circuits_input, commands_input], outputs=output)
    tf.keras.utils.plot_model(model, show_shapes=True, dpi=70, to_file="../images/cirquit-nn-model.png")
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='mse', metrics=['mse', r2_metric])
    return model, None, None

def encode_input_as_circuit(x: np.ndarray, qubits: list) -> cirq.Circuit:
    circuit = cirq.Circuit()
    for i, val in enumerate(x[:len(qubits)]):
        circuit.append([
            cirq.rx(val)(qubits[i]),
            cirq.ry(val)(qubits[i]),
            cirq.rz(val)(qubits[i])
        ])
    return circuit

def init_data(X_train, Y_train, X_test):
    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)

    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    Y_train_scaled = y_scaler.fit_transform(Y_train)

    n_samples, n_features = X_train.shape
    qubits = [cirq.GridQubit(0, i) for i in range(2)]

    controller = tf.keras.Sequential([
        tf.keras.Input(shape=(n_features,)),
        tf.keras.layers.Dense(N_NEURONS, activation=ACTIVATION),
        tf.keras.layers.Dense(6)
    ])

    datapoints_train = tfq.convert_to_tensor([
        encode_input_as_circuit(x, qubits) for x in X_train_scaled
    ])
    datapoints_test = tfq.convert_to_tensor([
        encode_input_as_circuit(x, qubits) for x in X_test_scaled
    ])
    commands_train = tf.convert_to_tensor(X_train_scaled.astype(np.float32))
    commands_test = tf.convert_to_tensor(X_test_scaled.astype(np.float32))

    return X_train_scaled, Y_train_scaled, X_test_scaled, qubits, n_samples, n_features, y_scaler, \
            controller, datapoints_train, datapoints_test, commands_train, commands_test

"""
Mean Squared Error on training samples: 0.3535
Mean Squared Error on test samples: 0.5340
Best model path: ../models/quantum_model_epoch00797_loss0.00034.weights.h5
Best model accuracy: 0.9900
 
Accuracy on training dataset: 99.71%
Accuracy on test dataset: 98.66%
Mean Squared Error on training samples: 0.0150
Mean Squared Error on test samples: 0.0971
 | Random Forest | Initialization. Total fit and predict time: 8.680685758590698 seconds
Accuracy on training dataset: 99.71%
Accuracy on test dataset: 98.66%
Mean Squared Error on training samples: 0.0150
Mean Squared Error on test samples: 0.0971
"""
def run_nn_quibit_on_parkinson(X_train, Y_train, X_test, Y_test, times, CV_times, r2, MSE):
    X_train_scaled, Y_train_scaled, X_test_scaled, qubits, n_samples, n_features, y_scaler, \
            controller, datapoints_train, datapoints_test, commands_train, commands_test = init_data(X_train, Y_train, X_test)
    model, _, _ = create_model(controller, qubits=qubits, input_dim=n_features)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='mse',
        metrics=['mse', r2_metric]
    )

    checkpoint_cb = ModelCheckpoint(
        filepath="../models/quantum_model_epoch{epoch:05d}_loss{loss:.5f}.weights.h5",
        monitor="mse",
        mode="min",
        save_best_only=True,
        save_weights_only=True,  # <-- This avoids serializing the model structure
        verbose=1
    )

    earlystop_cb = EarlyStopping(
        monitor="mse",
        mode="min",
        patience=15,
        restore_best_weights=True,
        verbose=1
    )

    start_time = time.time()
    history = model.fit(
        x=[datapoints_train, commands_train],
        y=Y_train_scaled,
        epochs=EPOCHS,
        verbose=1,
        callbacks=[checkpoint_cb] # earlystop_cb
    )
    times.append(time.time() - start_time)

    # Plot training loss
    plt.clf()
    plt.plot(history.history['loss'], label="Loss (MSE)")
    plt.plot(history.history['r2_metric'], label="R²", linestyle="--")
    plt.title("Quantum Model Training")
    plt.xlabel("Epochs")
    plt.ylabel("Metric Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("../images/quantum_nn_training_loss_r2.png")

"""
Mean Squared Error on training samples: 0.2742
Mean Squared Error on test samples: 0.4388
Best model path: ../models/quantum_model_epoch00271_loss0.00070.weights.h5
Best model accuracy: 0.9853
"""
def evaluate_models(X_train, Y_train, X_test, Y_test, times, CV_times, r2, MSE):
    X_train_scaled, Y_train_scaled, X_test_scaled, qubits, n_samples, n_features, y_scaler, \
            controller, datapoints_train, datapoints_test, commands_train, commands_test = init_data(X_train, Y_train, X_test)
    files: list = glob.glob("../models/*.h5", recursive=True)
    best_accuracy = 0
    best_model = None
    best_path = None
    for f in files:    
        # Rebuild same model structure
        model, _, _ = create_model(controller, qubits=qubits, input_dim=n_features)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='mse',
            metrics=['mse', r2_metric]
        )
        model.load_weights(f)
        result = compute_metrics(
            model=model,
            x_tr=[datapoints_train, commands_train],
            y_tr=Y_train,
            x_ts=[datapoints_test, commands_test],
            y_ts=Y_test,
            model_type='quantum',
            y_scaler=y_scaler
        )
        train_predict, test_predict, train_accuracy, test_accuracy, train_MSE, test_MSE = result
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model = model
            best_path = f
            print(f"New best model found: {f} with accuracy {test_accuracy:.4f}")
    print(f"Best model path: {best_path}")
    print(f"Best model accuracy: {best_accuracy:.4f}") 
    return best_model, best_accuracy, best_path
    
    #CV_times.append(0)
    #print(" | Quantum Neural Network | One Model. Training and Evaluation time: %.2f seconds" % fit_time)
    #r2.append(test_accuracy)
    #MSE.append(test_MSE)

def training_models(X_train, Y_train, X_test, Y_test):
    times = []
    CV_times = []
    r2 = []
    MSE = []
    # run_simple_nn_model(X_train, Y_train, X_test, Y_test, times, CV_times, r2, MSE)
    run_nn_quibit_on_parkinson(X_train, Y_train, X_test, Y_test, times, CV_times, r2, MSE)
    evaluate_models(X_train, Y_train, X_test, Y_test, times, CV_times, r2, MSE)

if __name__ == "__main__":
    X, Y, dataset = get_data()
    training_models(*calc_features(X, Y, dataset))
