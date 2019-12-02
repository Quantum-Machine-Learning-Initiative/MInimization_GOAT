import numpy as np
from numpy import kron, pi, real

from scipy.integrate import odeint, solve_ivp
from scipy.linalg import sqrtm, expm, eig, eigh, norm

import variationals
from variationals import *

import matplotlib.pyplot as plt

import time

import random
P_0 = np.array([[1,0],
                [0,0]]) # |0><0|
P_1 = np.array([[0,0],
                [0,1]]) # |1><1|

cswap = np.array([[1,0,0,0,0,0,0,0],
                  [0,1,0,0,0,0,0,0],
                  [0,0,1,0,0,0,0,0],
                  [0,0,0,1,0,0,0,0],
                  [0,0,0,0,1,0,0,0],
                  [0,0,0,0,0,0,1,0],
                  [0,0,0,0,0,1,0,0],
                  [0,0,0,0,0,0,0,1]])


def minimize_expectation_value(unitary_tensor_network, state_1, state_2, ref_state):
    expectation_values = []

    # create tensor network with random parameters
    unitary_tensor_network = variationals.heas_ansatz(qubits_number=3, layers_number=3)
    # extract the parameters and use as the initial ones
    initial_parameters = variationals.extract_parameters_from_tensor_network(unitary_tensor_network)

    def expectation_value(x):
        variationals.update_tensor_network(unitary_tensor_network, x)

        u_mat = variationals.tensor_network_to_matrix(qubits_number, unitary_tensor_network)

        hamiltonian = u_mat.dot(proj_f).dot(u_mat.conj().T)

        state = kron(kron(ref_state, state_1), state_2)

        expectation_value = (state.conj().T.dot(hamiltonian).dot(state)).real[0][0]

        expectation_values.append(expectation_value)

        return expectation_value

    optimization_result = minimize(expectation_value, initial_parameters, method='BFGS')

    variationals.update_tensor_network(unitary_tensor_network, optimization_result.x)

    u_mat = variationals.tensor_network_to_matrix(qubits_number, unitary_tensor_network)

    return u_mat, expectation_values

# P_\phi in the Hamiltonian
proj_f = kron(kron(P_1, I_np), I_np) + kron(kron(I_np, P_1), I_np) + kron(kron(I_np, I_np), P_1)

state_1 = np.array([[random.uniform(-1, 1) + 1j*random.uniform(-1, 1)] for i in range(2)])
state_2 = np.array([[random.uniform(-1, 1) + 1j*random.uniform(-1, 1)] for i in range(2)])

state_1 = state_1/norm(state_1)
state_2 = state_2/norm(state_2)

# ref_state = np.array([[1/sqrt(2)], [1/sqrt(2)]])
ref_state = np.array([[1], [0]])

res = minimize_expectation_value(unitary_tensor_network, state_1, state_2, ref_state)


def verification(u_mat, state_1, state_2, ref_state):
    overlap_start = abs(state_1.conj().T.dot(state_2))[0][0] ** 2

    hamiltonian = u_mat.dot(proj_f).dot(u_mat.conj().T)

    state = kron(kron(ref_state, state_1), state_2)

    #     state = hamiltonian.dot(state)
    state = u_mat.dot(state)
    #     state = kron(kron(H_np, I_np), I_np).dot(cswap).dot(kron(kron(H_np, I_np), I_np)).dot(state)

    #     d_matrix = np.outer(state, state.conj().T)
    d_matrix = state.dot(state.conj().T)

    measure_0_proj = kron(kron(P_0, I_np), I_np)
    #     measure_0_proj = kron(kron(I_np, P_0), I_np)
    #     measure_0_proj = kron(kron(I_np, I_np), P_0)

    overlap_finish = 2 * trace(d_matrix.dot(measure_0_proj)).real - 1

    print('Overlap start:', overlap_start)
    print('Overlap finish:', overlap_finish)

    return None  # abs(overlap_exact - overlap_calculated)

x_axis = np.arange(0, len(res[1]), 1)

plt.figure(figsize=(10, 5))
plt.plot(x_axis, res[1], color='purple', linewidth=2.5)
plt.hlines(y=0, xmin=0, xmax=len(res[1]), colors='green', linestyles='dashed', linewidth=2.5)
plt.xlabel('Minimization step', fontsize=16)
plt.ylabel('Energy', fontsize=16)
plt.show()

verification(res[0], state_1, state_2, ref_state)
print()
print('u=')
res[0]