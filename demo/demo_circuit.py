# Quantum Computers' based Early Diseases Detection

import os

import importlib, pkg_resources
importlib.reload(pkg_resources)

import tensorflow as tf
import tensorflow_quantum as tfq

import cirq
import sympy
import numpy as np

# visualization tools
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Sans'

from cirq.contrib.svg import SVGCircuit
import cairosvg

a, b = sympy.symbols('a b')

# Create two qubits
q0, q1 = cirq.GridQubit.rect(1, 2)

# Create a circuit on these qubits using the parameters you created above.
circuit = cirq.Circuit(cirq.rx(a).on(q0), cirq.ry(b).on(q1), cirq.CNOT(q0, q1))
svg_data = SVGCircuit(circuit)._repr_svg_()
svg_data = svg_data.replace('font-family="Arial"', 'font-family="DejaVu Sans"')
with open("images/circuit.svg", "w", encoding="utf-8") as f:
    f.write(svg_data)
cairosvg.svg2png(bytestring=svg_data.encode("utf-8"), write_to="images/circuit.png")

# Calculate a state vector with a=0.5 and b=-0.5.
resolver = cirq.ParamResolver({a: 0.5, b: -0.5})
output_state_vector = cirq.Simulator().simulate(circuit, resolver).final_state_vector
print("Output state vector: ", output_state_vector, flush=True)

z0 = cirq.Z(q0)
qubit_map = {q0: 0, q1: 1}
print("#1 Expectation from state vector: ", z0.expectation_from_state_vector(output_state_vector, qubit_map).real, flush=True)

z0x1 = 0.5 * z0 + cirq.X(q1)
print("#2 Expectation from state vector: ", z0x1.expectation_from_state_vector(output_state_vector, qubit_map).real, flush=True)

print("Rank 1 tensor containing 1 circuit.", flush=True)
circuit_tensor = tfq.convert_to_tensor([circuit])
print(circuit_tensor.shape, flush=True)
print(circuit_tensor.dtype, flush=True)

print("Rank 1 tensor containing 2 Pauli operators.", flush=True)
pauli_tensor = tfq.convert_to_tensor([z0, z0x1])
print(pauli_tensor.shape, flush=True)

batch_vals = np.array(np.random.uniform(0, 2 * np.pi, (5, 2)), dtype=float)
print("batch_vals: ", batch_vals, flush=True)

cirq_results = []
cirq_simulator = cirq.Simulator()
for vals in batch_vals:
    resolver = cirq.ParamResolver({a: vals[0], b: vals[1]})
    final_state_vector = cirq_simulator.simulate(circuit, resolver).final_state_vector
    cirq_results.append([
        z0.expectation_from_state_vector(final_state_vector, {
            q0: 0,
            q1: 1
        }).real
    ])

print(f"cirq batch results: {os.linesep} {np.array(cirq_results)}", flush=True)

# Parameters that the classical NN will feed values into.
control_params = sympy.symbols('theta_1 theta_2 theta_3')

# Create the parameterized circuit.
qubit = cirq.GridQubit(0, 0)
model_circuit = cirq.Circuit(
    cirq.rz(control_params[0])(qubit),
    cirq.ry(control_params[1])(qubit),
    cirq.rx(control_params[2])(qubit))

SVGCircuit(model_circuit)
svg_data = svg_data.replace('font-family="Arial"', 'font-family="DejaVu Sans"')
with open("images/circuit-2.svg", "w", encoding="utf-8") as f:
    f.write(svg_data)
cairosvg.svg2png(bytestring=svg_data.encode("utf-8"), write_to="images/circuit-2.png")


