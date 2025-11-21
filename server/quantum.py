import sympy
import cirq

import numpy as np

import tensorflow_quantum as tfq
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K

from sklearn.preprocessing import StandardScaler, MinMaxScaler

N_NEURONS = 75
ACTIVATION = 'relu'
LEARNING_RATE = 0.005

MODEL_PATH = "/home/vgarnaga/qc-edd/models/best/quantum_model_epoch00797_loss0.00034.weights.h5"

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

def init_data(X, x_scaler):
    X_scaled = x_scaler.transform(X)
    n_samples, n_features = X.shape
    qubits = [cirq.GridQubit(0, i) for i in range(2)]
    controller = tf.keras.Sequential([
        tf.keras.Input(shape=(n_features,)),
        tf.keras.layers.Dense(N_NEURONS, activation=ACTIVATION),
        tf.keras.layers.Dense(6)
    ])
    datapoints_train = tfq.convert_to_tensor([
        encode_input_as_circuit(x, qubits) for x in X_scaled
    ])
    commands = tf.convert_to_tensor(X_scaled.astype(np.float32))
    return X_scaled, qubits, n_samples, n_features, controller, datapoints_train, commands

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

    output = tf.keras.layers.Concatenate()(pqc_layers)

    model = tf.keras.Model(inputs=[circuits_input, commands_input], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='mse', metrics=['mse'])
    return model, None, None

def predict(x, x_scaler, y_scaler):
    print("Input shape: ", x.shape)
    x_scaled, qubits, n_samples, n_features, controller, datapoints_train, commands_train = init_data(x, x_scaler)
    model, _, _ = create_model(controller, qubits=qubits, input_dim=n_features)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='mse', metrics=['mse'])
    model.load_weights(MODEL_PATH)
    return y_scaler.inverse_transform(model.predict([datapoints_train, commands_train], verbose=0))
