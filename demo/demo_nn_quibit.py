# Quantum Computers' based Early Diseases Detection

from typing import Tuple

import os

import importlib, pkg_resources
importlib.reload(pkg_resources)

import tensorflow as tf
import tensorflow_quantum as tfq

import cirq
import sympy
import numpy as np

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Sans'

# It evaluates how well the hybrid model reconstructs the expected biological signature.
# Based on the value in `command_value` see how well you could prepare the full circuit to 
# have `desired_value` when taking expectation w.r.t. Z.
def check_error(command_values: list, desired_values: list, controller: tf.keras.Sequential, 
                model_circuit: cirq.Circuit, qubit: cirq.GridQubit, noisy_preparation: cirq.Circuit):
    params_to_prepare_output = controller(command_values).numpy()
    full_circuit = noisy_preparation + model_circuit
    cirq_simulator = cirq.Simulator()
    # Test how well you can prepare a state to get expectation the expectation value in `desired_values`
    for index in [0, 1]:
        state = cirq_simulator.simulate(full_circuit, {
            s: v
            for (s, v) in zip(control_params, params_to_prepare_output[index])
        }).final_state_vector
        
        expt = cirq.Z(qubit).expectation_from_state_vector(state, {
            qubit: 0
        }).real

        print(
            f"""For a desired output (expectation) of {desired_values[index]} with
            noisy preparation, the controller network found the following
            values for theta: {params_to_prepare_output[index]}
            Which gives an actual expectation of: {expt}{os.linesep}""")

# Create the parameterized circuit.
def create_quibit(control_params: enumerate) -> Tuple[cirq.Circuit, tf.keras.Sequential, cirq.GridQubit]:
    qubit = cirq.GridQubit(0, 0)
    model_circuit = cirq.Circuit(
        cirq.rz(control_params[0])(qubit),
        cirq.ry(control_params[1])(qubit),
        cirq.rx(control_params[2])(qubit))
    controller = tf.keras.Sequential(
        [tf.keras.layers.Dense(10, activation='elu'),
        tf.keras.layers.Dense(3)])
    # print(controller.summary(), flush=True)
    print(controller(tf.constant([[0.0], [1.0]])).numpy(), flush=True)
    return model_circuit, controller, qubit

# This input is the simulated miscalibration that the model will learn to correct.
def create_model(model_circuit: cirq.Circuit, controller: tf.keras.Sequential, qubit: cirq.GridQubit) -> \
                                                    Tuple[tf.keras.Model, np.ndarray, np.ndarray, tf.Tensor, cirq.Circuit]:
    circuits_input = tf.keras.Input(
        shape=(),
        # The circuit-tensor has dtype `tf.string`
        dtype=tf.string,
        name='circuits_input')
    # Commands will be either `0` or `1`, specifying the state to set the qubit to.
    commands_input = tf.keras.Input(shape=(1,), dtype=tf.dtypes.float32, name='commands_input')
    dense_2 = controller(commands_input)
    # TFQ layer for classically controlled circuits.
    expectation_layer = tfq.layers.ControlledPQC(
        model_circuit,
        # Observe Z
        operators=cirq.Z(qubit))
    expectation = expectation_layer([circuits_input, dense_2])
    # The full Keras model is built from our layers.
    model = tf.keras.Model(inputs=[circuits_input, commands_input], outputs=expectation)
    tf.keras.utils.plot_model(model, show_shapes=True, dpi=70, to_file="../images/model.png")
    # The command input values to the classical NN.
    commands = np.array([[0], [1]], dtype=np.float32)
    # The desired Z expectation value at output of quantum circuit.
    expected_outputs = np.array([[1], [-1]], dtype=np.float32)
    random_rotations = np.random.uniform(0, 2 * np.pi, 3)
    noisy_preparation = cirq.Circuit(
        cirq.rx(random_rotations[0])(qubit),
        cirq.ry(random_rotations[1])(qubit),
        cirq.rz(random_rotations[2])(qubit))
    datapoint_circuits = tfq.convert_to_tensor([noisy_preparation] * 2)  # Make two copied of this circuit
    print("datapoint_circuits.shape: ", datapoint_circuits.shape, flush=True)
    print(model([datapoint_circuits, commands]).numpy(), flush=True)
    return model, expected_outputs, commands, datapoint_circuits, noisy_preparation

def train_model(model: tf.keras.Model, expected_outputs: np.ndarray, commands: np.ndarray, 
                datapoint_circuits: tf.Tensor, noisy_preparation: cirq.Circuit) -> tf.keras.Model:
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)
    loss = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer=optimizer, loss=loss)
    history = model.fit(x=[datapoint_circuits, commands], y=expected_outputs, epochs=30, verbose=1)
    plt.plot(history.history['loss'])
    plt.title("Learning to Control a Qubit")
    plt.xlabel("Iterations")
    plt.ylabel("Error in Control")
    plt.savefig("../images/learning.png")
    check_error(commands, expected_outputs, controller, model_circuit, qubit, noisy_preparation)
    print(model([datapoint_circuits, commands]), flush=True)
    return model

if __name__=="__main__":
    # Parameters that the classical NN will feed values into.
    control_params: enumerate = sympy.symbols('theta_1 theta_2 theta_3')
    model_circuit, controller, qubit = create_quibit(control_params)
    model, expected_outputs, commands, datapoint_circuits, noisy_preparation = create_model(model_circuit, controller, qubit)
    model = train_model(model, expected_outputs, commands, datapoint_circuits, noisy_preparation)
