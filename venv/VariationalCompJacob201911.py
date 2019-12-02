# from projectq.ops import H, Measure, X, Y, Z, Swap, C, MatrixGate
# from projectq import MainEngine
# import projectq
import numpy as np
from numpy import kron
from scipy.linalg import expm
from numpy import ndarray
from scipy.optimize import minimize
import matplotlib.pyplot as plt

P_0 = np.array([[1, 0], [0, 0]])  # |0><0|
P_1 = np.array([[0, 0], [0, 1]])  # |1><1|
X_np = np.array([[0., 1.], [1., 0.]])  # X Pauli matrix
Y_np = np.array([[0., -1.j], [1.j, 0.]])  # Y Pauli matrix
Z_np = np.array([[1., 0.], [0., -1.]])  # Z Pauli matrix
S_np = np.array([[1., 0.], [0., 1.j]])
T_np = np.array([[1., 0.], [0., np.exp(1.j*np.pi/4)]])
I_np = np.array([[1., 0.], [0., 1.]])  # 2x2 identity matrix
H_np = (X_np + Z_np)/np.sqrt(2)  # Hadamard gate
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
CSWAR = 0.5 * kron((I_np + Z_np), kron(I_np, I_np)) + 0.5 * kron((I_np - Z_np), SWAP)
s0 = np.array([[1], [0]])
s1 = np.array([[0], [1]])
proj_1 = kron(I_np, I_np)
proj_1 = kron(P_1, proj_1)
proj_2 = kron(P_1, I_np)
proj_2 = kron(I_np, proj_2)
proj_3 = kron(I_np, P_1)
proj_3 = kron(I_np, proj_3)


def cx(qubits_number, qubits):
    cx_1 = P_0
    for i in range(qubits[0]):
        cx_1 = kron(I_np, cx_1)
    for i in range(qubits[0] + 1, qubits_number):
        cx_1 = kron(cx_1, I_np)
    if qubits[0] < qubits[1]:
        _op_1 = P_1
        _op_2 = X_np
        _q_1 = qubits[0]
        _q_2 = qubits[1]
    if qubits[0] > qubits[1]:
        _op_1 = X_np
        _op_2 = P_1
        _q_1 = qubits[1]
        _q_2 = qubits[0]
    cx_2 = _op_1
    for i in range(_q_1):
        cx_2 = kron(I_np, cx_2)
    for i in range(_q_1 + 1, _q_2):
        cx_2 = kron(cx_2, I_np)
    cx_2 = kron(cx_2, _op_2)
    for i in range(_q_2 + 1, qubits_number):
        cx_2 = kron(cx_2, I_np)
    cx = cx_1 + cx_2
    return cx


def Rx(a):
    return expm(-1j*X_np*a)


def Ry(a):
    return expm(-1j*Y_np*a)


def Rz(a):
    return expm(-1j*Z_np*a)


def U3(a, b, c):
    return Rz(c).dot(Rz(b)).dot(Rz(a))


def Projector(qubits_number, qubit_number):
    _projector = 1
    for i in range(qubit_number):
        _projector = kron(I_np, _projector)
    _projector = kron(P_1, _projector)
    for i in range(qubits_number - qubit_number - 1):
        _projector = kron(I_np, _projector)
    return _projector


def All_Proj():
    _All_Proj = np.add(Projector(3, 1), Projector(3, 2))
    _All_Proj = np.add(Projector(3, 0), _All_Proj)
    return _All_Proj


def Hamiltonian():
    _Hamiltonian = CSWAR
    _All_Proj = All_Proj()
    _Hamiltonian = _All_Proj.dot(_Hamiltonian)
    _Hamiltonian = CSWAR.conj().T.dot(_Hamiltonian)
    return _Hamiltonian


def U(_parameters, _q1, _q2, _q3):
    # 1st qubit
    _q1 = Rx(_parameters[0]).dot(_q1)
    _q1 = Rz(_parameters[1]).dot(_q1)
    _q1 = Rx(_parameters[2]).dot(_q1)
    # 2nd qubit
    _q2 = Rx(_parameters[3]).dot(_q2)
    _q2 = Rz(_parameters[4]).dot(_q2)
    _q2 = Rx(_parameters[5]).dot(_q2)
    # 3rd qubit
    _q3 = Rx(_parameters[6]).dot(_q3)
    _q3 = Rz(_parameters[7]).dot(_q3)
    _q3 = Rx(_parameters[8]).dot(_q3)
    _q_all = kron(_q2, _q3)
    _q_all = kron(_q1, _q_all)
    # controlled
    cx1 = cx(3, [0, 1])
    cx2 = cx(3, [1, 2])
    cx3 = cx(3, [2, 0])
    _q_all = cx1.dot(_q_all)
    _q_all = cx2.dot(_q_all)
    _q_all = cx3.dot(_q_all)
    return _q_all


def g_min_1(_parameters):  # Using several U gates
    q1 = [1, 0]
    q2 = [1, 0]
    q3 = [1, 0]
    q_all = U(_parameters, q1, q2, q3)
    q_all_proj = proj_1.dot(q_all)
    q_all_proj = proj_2.dot(q_all_proj)
    q_all_proj = proj_3.dot(q_all_proj)
    min_val.append(np.absolute(q_all.conj().T.dot(q_all_proj)))
    return np.absolute(q_all.conj().T.dot(q_all_proj))


def g_min_2(_parameters):  # _parameters[0] is theta, _parameters[1] is phi
    _q1 = H_np.dot([1, 0])
    _q2 = [1, 0]
    _q3 = [np.sin(_parameters[0]), np.exp(1.j * _parameters[1]) * np.cos(_parameters[0])]
    _q_all = kron(_q2, _q3)
    _q_all = kron(_q1, _q_all)
    min_val.append(np.absolute(_q_all.conj().T.dot(Hamiltonian().dot(_q_all))))
    return np.absolute(_q_all.conj().T.dot(Hamiltonian().dot(_q_all)))


parameters = np.array([0.1, 0.2])
min_val = []
res = minimize(g_min_2, parameters, method='L-BFGS-b', options={'gtol': 1e-8, 'disp': True})
print(res.x)
plt.plot(min_val)
plt.show()

