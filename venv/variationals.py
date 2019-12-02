import projectq
from projectq.ops import All, Measure, QubitOperator, TimeEvolution
from projectq.ops import ControlledGate, C, S, Sdag, X, Y, Z, H, Rx, Ry, Rz, CNOT, Swap
from projectq.meta import Control
from projectq import MainEngine
from projectq.backends import Simulator

import numpy as np
import random
from numpy import kron, trace, shape, real, sqrt, exp, cos, sin, pi
from scipy.linalg import expm, eigh, norm, det
from scipy.optimize import minimize

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

from math import ceil

import time

import copy


# -i*CNOT
# very useful sometimes
def myCNOT(qubits):

    CN = 0.5*(QubitOperator("") - QubitOperator("Z" + str(qubits[0])))*QubitOperator("X" + str(qubits[1])) + 0.5*(QubitOperator("") + QubitOperator("Z" + str(qubits[0])))
    return TimeEvolution(np.pi/2, CN)


#####################
# helpful functions #
#####################


def kron_chain(matrices):

    operator = matrices[0]

    for i in range(1, len(matrices)):
        operator = kron(operator, matrices[i])

    return operator


def dot_chain(matrices):

    operator = matrices[0]

    for i in range(1, len(matrices)):
        operator = np.dot(operator, matrices[i])

    return operator


def commutator(op_1, op_2):
    return op_1.dot(op_2) - op_2.dot(op_1)


def eigenvectorness(target_vector, matrix):
    return abs((target_vector.conjugate().transpose()).dot(matrix.dot(target_vector)))[0][0]


def overlap(vector_1, vector_2):
    return abs((vector_1.conjugate().transpose()).dot(vector_2))[0]


def two_qubit_entanglement_measure(state_vector):
    return 2*abs(state_vector[0]*state_vector[3] - state_vector[1]*state_vector[2])[0]


#############
# unitaries #
#############

# ProjectQ #

def unitary_operator(qubits_number, operator_name, qubits, parameters, derivatives=[], order=1, daggered=False):
    """
    Input:
        name of a unitary,
        parameters of this unitary,
        initial state,
        qubits to act on
    Return:
        list of unitary operators (ProjectQ objects) which should be applied one by one
        from the begining of the list to its end
    """

    # sign of the unitary evolutions
    # inversed for the daggered gates
    sign = 1
    if daggered == True:
        sign = -1

    unitaries_list = []

  ####################
  # simple rotations #
  ####################

    if operator_name == 'r_x':

        # performs unitary evolution e^{-i*parameter*operator} of the state
        unitaries_list.append(TimeEvolution(sign*parameters[0], QubitOperator("X"+str(qubits[0]))))
        for i in range(derivatives.count(0)):
            unitaries_list.append(-1j*sign*QubitOperator("X"+str(qubits[0])))

        return unitaries_list

    if operator_name == 'r_y':

        unitaries_list.append(TimeEvolution(sign*parameters[0], QubitOperator("Y"+str(qubits[0]))))
        for i in range(derivatives.count(0)):
            unitaries_list.append(-1j*sign*QubitOperator("Y"+str(qubits[0])))

        return unitaries_list

    if operator_name == 'r_z':

        unitaries_list.append(TimeEvolution(sign*parameters[0], QubitOperator("Z"+str(qubits[0]))))
        for i in range(derivatives.count(0)):
            unitaries_list.append(-1j*sign*QubitOperator("Z"+str(qubits[0])))

        return unitaries_list

  ###################
  # controlled gate #
  ###################

    if operator_name == 'cx':
    
        unitaries_list.append(myCNOT(qubits))
        unitaries_list.append(1j*QubitOperator(""))

        return unitaries_list

  ##############
  # SU(2) gate #
  ##############

    if operator_name == 'su2':

        unitaries_list.append(TimeEvolution(sign*parameters[0], QubitOperator("Z"+str(qubits[0]))))
        for i in range(derivatives.count(0)):
            unitaries_list.append(-1j*sign*QubitOperator("Z"+str(qubits[0])))
        unitaries_list.append(TimeEvolution(sign*parameters[1], QubitOperator("X"+str(qubits[0]))))
        for i in range(derivatives.count(1)):
            unitaries_list.append(-1j*sign*QubitOperator("X"+str(qubits[0])))
        unitaries_list.append(TimeEvolution(sign*parameters[2], QubitOperator("Z"+str(qubits[0]))))
        for i in range(derivatives.count(2)):
            unitaries_list.append(-1j*sign*QubitOperator("Z"+str(qubits[0])))

        # inverse the order of operators in list for the daggered gates
        if daggered == False:
            return unitaries_list
        if daggered == True:
            return unitaries_list[::-1]

  ##############
  # SU(4) gate #
  ##############

    if operator_name == 'su4':

        unitaries_list.append(TimeEvolution(sign*parameters[0], QubitOperator("Z"+str(qubits[0]))))
        for i in range(derivatives.count(0)):
            unitaries_list.append(-1j*sign*QubitOperator("Z"+str(qubits[0])))
        unitaries_list.append(TimeEvolution(sign*parameters[1], QubitOperator("X"+str(qubits[0]))))
        for i in range(derivatives.count(1)):
            unitaries_list.append(-1j*sign*QubitOperator("X"+str(qubits[0])))
        unitaries_list.append(TimeEvolution(sign*parameters[2], QubitOperator("Z"+str(qubits[0]))))
        for i in range(derivatives.count(2)):
            unitaries_list.append(-1j*sign*QubitOperator("Z"+str(qubits[0])))

        unitaries_list.append(TimeEvolution(sign*parameters[3], QubitOperator("Z"+str(qubits[1]))))
        for i in range(derivatives.count(3)):
            unitaries_list.append(-1j*sign*QubitOperator("Z"+str(qubits[1])))
        unitaries_list.append(TimeEvolution(sign*parameters[4], QubitOperator("X"+str(qubits[1]))))
        for i in range(derivatives.count(4)):
            unitaries_list.append(-1j*sign*QubitOperator("X"+str(qubits[1])))
        unitaries_list.append(TimeEvolution(sign*parameters[5], QubitOperator("Z"+str(qubits[1]))))
        for i in range(derivatives.count(5)):
            unitaries_list.append(-1j*sign*QubitOperator("Z"+str(qubits[1])))

        unitaries_list.append(TimeEvolution(sign, parameters[6]*QubitOperator("X"+str(qubits[0]) + " X"+str(qubits[1])) + parameters[7]*QubitOperator("Y"+str(qubits[0]) + " Y"+str(qubits[1])) + parameters[8]*QubitOperator("Z"+str(qubits[0]) + " Z"+str(qubits[1]))))
        for i in range(derivatives.count(6)):
            unitaries_list.append(-1j*sign*QubitOperator("X"+str(qubits[0]) + " X"+str(qubits[1])))
        for i in range(derivatives.count(7)):
            unitaries_list.append(-1j*sign*QubitOperator("Y"+str(qubits[0]) + " Y"+str(qubits[1])))
        for i in range(derivatives.count(8)):
            unitaries_list.append(-1j*sign*QubitOperator("Z"+str(qubits[0]) + " Z"+str(qubits[1])))

        unitaries_list.append(TimeEvolution(sign*parameters[9], QubitOperator("Z"+str(qubits[0]))))
        for i in range(derivatives.count(9)):
            unitaries_list.append(-1j*sign*QubitOperator("Z"+str(qubits[0])))
        unitaries_list.append(TimeEvolution(sign*parameters[10], QubitOperator("X"+str(qubits[0]))))
        for i in range(derivatives.count(10)):
            unitaries_list.append(-1j*sign*QubitOperator("X"+str(qubits[0])))
        unitaries_list.append(TimeEvolution(sign*parameters[11], QubitOperator("Z"+str(qubits[0]))))
        for i in range(derivatives.count(11)):
            unitaries_list.append(-1j*sign*QubitOperator("Z"+str(qubits[0])))

        unitaries_list.append(TimeEvolution(sign*parameters[12], QubitOperator("Z"+str(qubits[1]))))
        for i in range(derivatives.count(12)):
            unitaries_list.append(-1j*sign*QubitOperator("Z"+str(qubits[1])))
        unitaries_list.append(TimeEvolution(sign*parameters[13], QubitOperator("X"+str(qubits[1]))))
        for i in range(derivatives.count(13)):
            unitaries_list.append(-1j*sign*QubitOperator("X"+str(qubits[1])))
        unitaries_list.append(TimeEvolution(sign*parameters[14], QubitOperator("Z"+str(qubits[1]))))
        for i in range(derivatives.count(14)):
            unitaries_list.append(-1j*sign*QubitOperator("Z"+str(qubits[1])))

        # inverse the order of operators in list for the daggered gates
        if daggered == False:
            return unitaries_list
        if daggered == True:
            return unitaries_list[::-1]


  ###############
  # other gates #
  ###############

    if operator_name == 'rank1':
        
        d_0 = derivatives.count(0)
        d_1 = derivatives.count(1)
        d_2 = derivatives.count(2)
        d_3 = derivatives.count(3)
        d_4 = derivatives.count(4)
        d_5 = derivatives.count(5)
        
        # su2 gate on qubit-0
        unitaries_list.append(TimeEvolution(sign*(parameters[0] + d_0*pi/2), QubitOperator("X"+str(qubits[0]))))
        unitaries_list.append(TimeEvolution(sign*(parameters[1] + d_1*pi/2), QubitOperator("Y"+str(qubits[0]))))
        unitaries_list.append(TimeEvolution(sign*(parameters[2] + d_2*pi/2), QubitOperator("X"+str(qubits[0]))))

        # su2 gate on qubit-1
        unitaries_list.append(TimeEvolution(sign*(parameters[3] + d_3*pi/2), QubitOperator("X"+str(qubits[1]))))
        unitaries_list.append(TimeEvolution(sign*(parameters[4] + d_4*pi/2), QubitOperator("Y"+str(qubits[1]))))
        unitaries_list.append(TimeEvolution(sign*(parameters[5] + d_5*pi/2), QubitOperator("X"+str(qubits[1]))))
        
        if daggered == False:
            return unitaries_list
        if daggered == True:
            return unitaries_list[::-1]
        
        ### old version ###
#         # u_3 gate on qubit-1
#         unitaries_list.append(TimeEvolution(sign*parameters[0], QubitOperator("Z"+str(qubits[0]))))
#         for i in range(derivatives.count(0)):
#             unitaries_list.append(-1j*sign*QubitOperator("Z"+str(qubits[0])))
#         unitaries_list.append(TimeEvolution(sign*parameters[1], QubitOperator("X"+str(qubits[0]))))
#         for i in range(derivatives.count(1)):
#             unitaries_list.append(-1j*sign*QubitOperator("X"+str(qubits[0])))
#         unitaries_list.append(TimeEvolution(sign*parameters[2], QubitOperator("Z"+str(qubits[0]))))
#         for i in range(derivatives.count(2)):
#             unitaries_list.append(-1j*sign*QubitOperator("Z"+str(qubits[0])))

#         # u_3 gate on qubit-1
#         unitaries_list.append(TimeEvolution(sign*parameters[3], QubitOperator("Z"+str(qubits[1]))))
#         for i in range(derivatives.count(3)):
#             unitaries_list.append(-1j*sign*QubitOperator("Z"+str(qubits[1])))
#         unitaries_list.append(TimeEvolution(sign*parameters[4], QubitOperator("X"+str(qubits[1]))))
#         for i in range(derivatives.count(4)):
#             unitaries_list.append(-1j*sign*QubitOperator("X"+str(qubits[1])))
#         unitaries_list.append(TimeEvolution(sign*parameters[5], QubitOperator("Z"+str(qubits[1]))))
#         for i in range(derivatives.count(5)):
#             unitaries_list.append(-1j*sign*QubitOperator("Z"+str(qubits[1])))

#         if daggered == False:
#             return unitaries_list
#         if daggered == True:
#             return unitaries_list[::-1]

    if operator_name == 'rank2':

        d_0 = derivatives.count(0)
        d_1 = derivatives.count(1)
        d_2 = derivatives.count(2)
        d_3 = derivatives.count(3)
        d_4 = derivatives.count(4)
        d_5 = derivatives.count(5)  
        d_6 = derivatives.count(6) 
        
        # r_y rotation of the qubit-0
        unitaries_list.append(TimeEvolution(sign*(parameters[0] + d_0*pi/2), QubitOperator("Y"+str(qubits[0]))))

        # CNOT from qubit-0 to qubit-1
        unitaries_list.append(myCNOT(qubits))
        # to compensate the phase of myCNOT gate
        unitaries_list.append(1j*QubitOperator(""))

        # su2 gate on qubit-0
        unitaries_list.append(TimeEvolution(sign*(parameters[1] + d_1*pi/2), QubitOperator("X"+str(qubits[0]))))
        unitaries_list.append(TimeEvolution(sign*(parameters[2] + d_2*pi/2), QubitOperator("Y"+str(qubits[0]))))
        unitaries_list.append(TimeEvolution(sign*(parameters[3] + d_3*pi/2), QubitOperator("X"+str(qubits[0]))))

        # su2 gate on qubit-1
        unitaries_list.append(TimeEvolution(sign*(parameters[4] + d_4*pi/2), QubitOperator("X"+str(qubits[1]))))
        unitaries_list.append(TimeEvolution(sign*(parameters[5] + d_5*pi/2), QubitOperator("Y"+str(qubits[1]))))
        unitaries_list.append(TimeEvolution(sign*(parameters[6] + d_6*pi/2), QubitOperator("X"+str(qubits[1]))))

        if daggered == False:
            return unitaries_list
        if daggered == True:
            return unitaries_list[::-1]

    if operator_name == 'ising':

        d_0 = derivatives.count(0)
        d_1 = derivatives.count(1)
        d_2 = derivatives.count(2)
        d_3 = derivatives.count(3)
        d_4 = derivatives.count(4)
        d_5 = derivatives.count(5)        
        
        unitaries_list.append(TimeEvolution(sign*(parameters[0] + d_0*pi/2), QubitOperator("X"+str(qubits[0]))))
        unitaries_list.append(TimeEvolution(sign*(parameters[1] + d_1*pi/2), QubitOperator("X"+str(qubits[1]))))
        unitaries_list.append(TimeEvolution(sign*(parameters[2] + d_2*pi/2), QubitOperator("Z"+str(qubits[0]) + " Z"+str(qubits[1]))))
        unitaries_list.append(TimeEvolution(sign*(parameters[3] + d_3*pi/2), QubitOperator("Z"+str(qubits[0]))))
        unitaries_list.append(TimeEvolution(sign*(parameters[4] + d_4*pi/2), QubitOperator("Z"+str(qubits[1]))))

        if daggered == False:
            return unitaries_list
        if daggered == True:
            return unitaries_list[::-1]

# NumPy #

s0 = np.array([[1],
               [0]]) # |0>
s1 = np.array([[0],
               [1]]) # |0>
P_0 = np.array([[1,0],[0,0]]) # |0><0|
P_1 = np.array([[0,0],[0,1]]) # |1><1|
X_np = np.array([[0.,1.],
                 [1.,0.]]) # X Pauli matrix
Y_np = np.array([[0.,-1.j],
                 [1.j, 0.]]) # Y Pauli matrix
Z_np = np.array([[1., 0.],
                 [0.,-1.]]) # Z Pauli matrix
I_np = np.array([[1.,0.],
                 [0.,1.]]) # 2x2 identity matrix
H_np = (X_np + Z_np)/np.sqrt(2) # Hadamard gate


def unitary_matrix(operator_name, qubits_number, qubits, parameters, daggered=False, derivatives=[]):
    """
    Input:
        name of a unitary,
        parameters of this unitary,
        qubits number,
        qubits to act on
    Return:
        NumPy matrix representation of a unitary operator
    """
  ###################
  # basic rotations #
  ###################

    if operator_name == 'r_x':

        operator = expm(-1.j*parameters[0]*X_np)
        for i in range(derivatives.count(0)):
            operator = operator.dot(-1.j*X_np)

        for j in range(qubits[0]):
            operator = kron(I_np, operator)
        for j in range(qubits[0]+1, qubits_number):
            operator = kron(operator, I_np)

        if daggered == False:
            return operator
        if daggered == True:
            return operator.conjugate().transpose()

    if operator_name == 'r_y':

        operator = expm(-1.j*parameters[0]*Y_np)
        for i in range(derivatives.count(0)):
            operator = operator.dot(-1.j*Y_np)

        for j in range(qubits[0]):
            operator = kron(I_np, operator)
        for j in range(qubits[0]+1, qubits_number):
            operator = kron(operator, I_np)

        if daggered == False:
            return operator
        if daggered == True:
            return operator.conjugate().transpose()

    if operator_name == 'r_z':

        operator = expm(-1.j*parameters[0]*Z_np)
        for i in range(derivatives.count(0)):
            operator = operator.dot(-1.j*Z_np)

        for j in range(qubits[0]):
            operator = kron(I_np, operator)
        for j in range(qubits[0]+1, qubits_number):
            operator = kron(operator, I_np)

        if daggered == False:
            return operator
        if daggered == True:
            return operator.conjugate().transpose()

  ##########################
  # controlled basic gates #
  ##########################
      
    if operator_name == 'cx':

        cx_1 = P_0
        for i in range(qubits[0]):
            cx_1 = kron(I_np, cx_1)
        for i in range(qubits[0]+1, qubits_number):
            cx_1 = kron(cx_1, I_np)

        if qubits[0] < qubits[1]:
            op_1 = P_1
            op_2 = X_np
            q_1 = qubits[0]
            q_2 = qubits[1]
        if qubits[0] > qubits[1]:
            op_1 = X_np
            op_2 = P_1
            q_1 = qubits[1]
            q_2 = qubits[0]
        cx_2 = op_1
        for i in range(q_1):
            cx_2 = kron(I_np, cx_2)
        for i in range(q_1+1, q_2):
            cx_2 = kron(cx_2, I_np)
        cx_2 = kron(cx_2, op_2)
        for i in range(q_2+1, qubits_number):
            cx_2 = kron(cx_2, I_np)

        cx = cx_1 + cx_2

        return cx
    
    if operator_name == 'cz':

        cz_1 = P_0
        for i in range(qubits[0]):
            cz_1 = kron(I_np, cz_1)
        for i in range(qubits[0]+1, qubits_number):
            cz_1 = kron(cz_1, I_np)

        if qubits[0] < qubits[1]:
            op_1 = P_1
            op_2 = Z_np
            q_1 = qubits[0]
            q_2 = qubits[1]
        if qubits[0] > qubits[1]:
            op_1 = Z_np
            op_2 = P_1
            q_1 = qubits[1]
            q_2 = qubits[0]
        cz_2 = op_1
        for i in range(q_1):
            cz_2 = kron(I_np, cz_2)
        for i in range(q_1+1, q_2):
            cz_2 = kron(cz_2, I_np)
        cz_2 = kron(cz_2, op_2)
        for i in range(q_2+1, qubits_number):
            cz_2 = kron(cz_2, I_np)

        cz = cz_1 + cz_2

        return cz
    
  ##############################
  # controlled basic rotations #
  ##############################

    if operator_name == 'cr_x':

        r_x = expm(-1.j*parameters[0]*X_np)
        for i in range(derivatives.count(0)):
            r_x = r_x.dot(-1.j*X_np)

        cr_x_1 = P_0
        for i in range(qubits[0]):
            cr_x_1 = kron(I_np, cr_x_1)
        for i in range(qubits[0]+1, qubits_number):
            cr_x_1 = kron(cr_x_1, I_np)

        if qubits[0] < qubits[1]:
            op_1 = P_1
            op_2 = r_x
            q_1 = qubits[0]
            q_2 = qubits[1]
        if qubits[0] > qubits[1]:
            op_1 = r_x
            op_2 = P_1
            q_1 = qubits[1]
            q_2 = qubits[0]
        cr_x_2 = op_1
        for i in range(q_1):
            cr_x_2 = kron(I_np, cr_x_2)
        for i in range(q_1+1, q_2):
            cr_x_2 = kron(cr_x_2, I_np)
        cr_x_2 = kron(cr_x_2, op_2)
        for i in range(q_2+1, qubits_number):
            cr_x_2 = kron(cr_x_2, I_np)

        cr_x = cr_x_1 + cr_x_2

        if daggered == False:
            return cr_x
        if daggered == True:
            return cr_x.conjugate().transpose()


    if operator_name == 'cr_y':

        d_0 = derivatives.count(0)
        
        if d_0 == 0:
            cr_y_1 = P_0
            for i in range(qubits[0]):
                cr_y_1 = kron(I_np, cr_y_1)
            for i in range(qubits[0]+1, qubits_number):
                cr_y_1 = kron(cr_y_1, I_np)
        if d_0 != 0:
            cr_y_1 = np.zeros((2**qubits_number, 2**qubits_number))

        r_y = expm(-1.j*parameters[0]*Y_np)
        for i in range(derivatives.count(0)):
            r_y = r_y.dot(-1.j*Y_np)
        if qubits[0] < qubits[1]:
            op_1 = P_1
            op_2 = r_y
            q_1 = qubits[0]
            q_2 = qubits[1]
        if qubits[0] > qubits[1]:
            op_1 = r_y
            op_2 = P_1
            q_1 = qubits[1]
            q_2 = qubits[0]
        cr_y_2 = op_1
        for i in range(q_1):
            cr_y_2 = kron(I_np, cr_y_2)
        for i in range(q_1+1, q_2):
            cr_y_2 = kron(cr_y_2, I_np)
        cr_y_2 = kron(cr_y_2, op_2)
        for i in range(q_2+1, qubits_number):
            cr_y_2 = kron(cr_y_2, I_np)

        cr_y = cr_y_1 + cr_y_2

        if daggered == False:
            return cr_y
        if daggered == True:
            return cr_y.conjugate().transpose()


    if operator_name == 'cr_z':

        r_z = expm(-1.j*parameters[0]*Z_np)
        for i in range(derivatives.count(0)):
            r_z = r_z.dot(-1.j*Z_np)

        cr_z_1 = P_0
        for i in range(qubits[0]):
            cr_z_1 = kron(I_np, cr_z_1)
        for i in range(qubits[0]+1, qubits_number):
            cr_z_1 = kron(cr_z_1, I_np)

        if qubits[0] < qubits[1]:
            op_1 = P_1
            op_2 = r_z
            q_1 = qubits[0]
            q_2 = qubits[1]
        if qubits[0] > qubits[1]:
            op_1 = r_z
            op_2 = P_1
            q_1 = qubits[1]
            q_2 = qubits[0]
        cr_z_2 = op_1
        for i in range(q_1):
            cr_z_2 = kron(I_np, cr_z_2)
        for i in range(q_1+1, q_2):
            cr_z_2 = kron(cr_z_2, I_np)
        cr_z_2 = kron(cr_z_2, op_2)
        for i in range(q_2+1, qubits_number):
            cr_z_2 = kron(cr_z_2, I_np)

        cr_z = cr_z_1 + cr_z_2

        if daggered == False:
            return cr_z
        if daggered == True:
            return cr_z.conjugate().transpose()


  ##############
  # SU(2) gate #
  ##############

    if operator_name == 'su2':

        d_0 = derivatives.count(0)
        d_1 = derivatives.count(1)
        d_2 = derivatives.count(2)       
        
        op_1 = expm(-1.j/2 * (2*parameters[2] + d_2*pi/2) * X_np)
        op_2 = expm(-1.j/2 * (2*parameters[0] + d_0*pi/2) * Y_np)
        op_3 = expm(-1.j/2 * (2*parameters[1] + d_1*pi/2) * X_np)
        
        operator = dot_chain([op_3, op_2, op_1])

        for j in range(qubits[0]):
            operator = kron(I_np, operator)
        for j in range(qubits[0]+1, qubits_number):
            operator = kron(operator, I_np)

        if daggered == False:
            return operator
        if daggered == True:
            return operator.conjugate().transpose()
        
    ### unused versions ###
#         c_11 = 2**(-d_0) * cos((parameters[0] + d_0*pi)/2)
#         c_12 = -exp(1j*(parameters[2] + d_2*pi)) * 2**(-d_0) * sin((parameters[0] + d_0*pi)/2)
#         c_21 =  exp(1j*(parameters[1] + d_1*pi)) * 2**(-d_0) * sin((parameters[0] + d_0*pi)/2)
#         c_22 =  exp(1j*(parameters[1] + parameters[2] + d_1*pi + d_2*pi)) * 2**(-d_0) * cos((parameters[0] + d_0*pi)/2)

#         c_11 =  cos(1/2 * (2*parameters[0] + d_0*pi/2))
#         c_12 = -exp(1j * (2*parameters[2] + d_2*pi/2)) * sin(1/2 * (2*parameters[0] + d_0*pi/2))
#         c_21 =  exp(1j * (2*parameters[1] + d_1*pi/2)) * sin(1/2 * (2*parameters[0] + d_0*pi/2))
#         c_22 =  exp(1j * (2*parameters[1] + 2*parameters[2] + d_1*pi/2 + d_2*pi/2)) * cos(1/2 * (2*parameters[0] + d_0*pi/2))
        
#         operator = np.array([[c_11, c_12],
#                              [c_21, c_22]])
        
        
    ### old version ###
#         op_1 = expm(-1.j*parameters[0]*X_np)
#         for i in range(derivatives.count(0)):
#             op_1 = op_1.dot(-1.j*X_np)
#         op_2 = expm(-1.j*parameters[1]*Z_np)
#         for i in range(derivatives.count(1)):
#             op_2 = op_2.dot(-1.j*Z_np)
#         op_3 = expm(-1.j*parameters[2]*X_np)
#         for i in range(derivatives.count(2)):
#             op_3 = op_3.dot(-1.j*X_np)

#         operator = dot_chain([op_3, op_2, op_1])

#         for j in range(qubits[0]):
#             operator = kron(I_np, operator)
#         for j in range(qubits[0]+1, qubits_number):
#             operator = kron(operator, I_np)

#         if daggered == False:
#             return operator
#         if daggered == True:
#             return operator.conjugate().transpose()

  ##############
  # SU(4) gate #
  ##############

    if operator_name == 'su4':

        XX = X_np
        for j in range(qubits[0]):
            XX = kron(I_np, XX)
        for j in range(qubits[0]+1, qubits[1]):
            XX = kron(XX, I_np)
        XX = kron(XX, X_np)
        for j in range(qubits[1]+1, qubits_number):
            XX = kron(XX, I_np)

        YY = Y_np
        for j in range(qubits[0]):
            YY = kron(I_np, YY)
        for j in range(qubits[0]+1, qubits[1]):
            YY = kron(YY, I_np)
        YY = kron(YY, Y_np)
        for j in range(qubits[1]+1, qubits_number):
            YY = kron(YY, I_np)

        ZZ = Z_np
        for j in range(qubits[0]):
            ZZ = kron(I_np, ZZ)
        for j in range(qubits[0]+1, qubits[1]):
            ZZ = kron(ZZ, I_np)
        ZZ = kron(ZZ, Z_np)
        for j in range(qubits[1]+1, qubits_number):
            ZZ = kron(ZZ, I_np)

        op_1 = expm(-1.j*parameters[0]*Z_np)
        for i in range(derivatives.count(0)):
            op_1 = op_1.dot(-1.j*Z_np)
        op_2 = expm(-1.j*parameters[1]*X_np)
        for i in range(derivatives.count(1)):
            op_2 = op_2.dot(-1.j*X_np)
        op_3 = expm(-1.j*parameters[2]*Z_np)
        for i in range(derivatives.count(2)):
            op_3 = op_3.dot(-1.j*Z_np)
        su2_1 = dot_chain([op_3, op_2, op_1])

        op_1 = expm(-1.j*parameters[3]*Z_np)
        for i in range(derivatives.count(3)):
            op_1 = op_1.dot(-1.j*Z_np)
        op_2 = expm(-1.j*parameters[4]*X_np)
        for i in range(derivatives.count(4)):
            op_2 = op_2.dot(-1.j*X_np)
        op_3 = expm(-1.j*parameters[5]*Z_np)
        for i in range(derivatives.count(5)):
            op_3 = op_3.dot(-1.j*Z_np)
        su2_2 = dot_chain([op_3, op_2, op_1])

        term_1 = su2_1
        for j in range(qubits[0]):
            term_1 = kron(I_np, term_1)
        for j in range(qubits[0]+1, qubits[1]):
            term_1 = kron(term_1, I_np)
        term_1 = kron(term_1, su2_2)
        for j in range(qubits[1]+1, qubits_number):
            term_1 = kron(term_1, I_np)

        #su2_1 = unitary_matrix(operator_name='su2', qubits_number=qubits_number, qubits=[qubits[0]], parameters=parameters[:3], daggered=daggered, derivatives=derivatives)
        #su2_2 = unitary_matrix(operator_name='su2', qubits_number=qubits_number, qubits=[qubits[1]], parameters=parameters[3:6], daggered=daggered, derivatives=list(np.array(derivatives)-3))

        op = expm(-1j*(parameters[6]*XX + parameters[7]*YY + parameters[8]*ZZ))
        for i in range(derivatives.count(6)):
            op = op.dot(-1.j*XX)
        for i in range(derivatives.count(7)):
            op = op.dot(-1.j*YY)
        for i in range(derivatives.count(8)):
            op = op.dot(-1.j*ZZ)

        op_1 = expm(-1.j*parameters[9]*Z_np)
        for i in range(derivatives.count(9)):
            op_1 = op_1.dot(-1.j*Z_np)
        op_2 = expm(-1.j*parameters[10]*X_np)
        for i in range(derivatives.count(10)):
            op_2 = op_2.dot(-1.j*X_np)
        op_3 = expm(-1.j*parameters[11]*Z_np)
        for i in range(derivatives.count(11)):
            op_3 = op_3.dot(-1.j*Z_np)
        su2_3 = dot_chain([op_3, op_2, op_1])

        op_1 = expm(-1.j*parameters[12]*Z_np)
        for i in range(derivatives.count(12)):
            op_1 = op_1.dot(-1.j*Z_np)
        op_2 = expm(-1.j*parameters[13]*X_np)
        for i in range(derivatives.count(13)):
            op_2 = op_2.dot(-1.j*X_np)
        op_3 = expm(-1.j*parameters[14]*Z_np)
        for i in range(derivatives.count(14)):
            op_3 = op_3.dot(-1.j*Z_np)
        su2_4 = dot_chain([op_3, op_2, op_1])

        term_2 = su2_3
        for j in range(qubits[0]):
            term_2 = kron(I_np, term_2)
        for j in range(qubits[0]+1, qubits[1]):
            term_2 = kron(term_2, I_np)
        term_2 = kron(term_2, su2_4)
        for j in range(qubits[1]+1, qubits_number):
            term_2 = kron(term_2, I_np)

        #su2_3 = unitary_matrix(operator_name='su2', qubits_number=qubits_number, qubits=[qubits[0]], parameters=parameters[9:12], daggered=daggered, derivatives=list(np.array(derivatives)-9))
        #su2_4 = unitary_matrix(operator_name='su2', qubits_number=qubits_number, qubits=[qubits[1]], parameters=parameters[12:15], daggered=daggered, derivatives=list(np.array(derivatives)-12))

        #operator = dot_chain([su2_1, su2_2, op, su2_3, su2_4])
        #print(np.shape(op))
        operator = dot_chain([term_1, op, term_2])

        if daggered == False:
            return operator
        if daggered == True:
            return operator.conjugate().transpose()

  ##############
  # SU(8) gate #
  ##############

    if operator_name == 'su8':

        XXZ = X_np
        for j in range(qubits[0]):
            XXZ = kron(I_np, XXZ)
        for j in range(qubits[0]+1, qubits[1]):
            XXZ = kron(XXZ, I_np)
        XXZ = kron(XXZ, X_np)
        for j in range(qubits[1]+1, qubits[2]):
            XXZ = kron(XXZ, I_np)
        XXZ = kron(XXZ, Z_np)
        for j in range(qubits[2]+1, qubits_number):
            XXZ = kron(XXZ, I_np)

        YYZ = Y_np
        for j in range(qubits[0]):
            YYZ = kron(I_np, YYZ)
        for j in range(qubits[0]+1, qubits[1]):
            YYZ = kron(YYZ, I_np)
        YYZ = kron(YYZ, Y_np)
        for j in range(qubits[1]+1, qubits[2]):
            YYZ = kron(YYZ, I_np)
        YYZ = kron(YYZ, Z_np)
        for j in range(qubits[2]+1, qubits_number):
            YYZ = kron(YYZ, I_np)

        ZZZ = Z_np
        for j in range(qubits[0]):
            ZZZ = kron(I_np, ZZZ)
        for j in range(qubits[0]+1, qubits[1]):
            ZZZ = kron(ZZZ, I_np)
        ZZZ = kron(ZZZ, Z_np)
        for j in range(qubits[1]+1, qubits[2]):
            ZZZ = kron(ZZZ, I_np)
        ZZZ = kron(ZZZ, Z_np)
        for j in range(qubits[2]+1, qubits_number):
            ZZZ = kron(ZZZ, I_np)

        XXX = X_np
        for j in range(qubits[0]):
            XXX = kron(I_np, XXX)
        for j in range(qubits[0]+1, qubits[1]):
            XXX = kron(XXX, I_np)
        XXX = kron(XXX, X_np)
        for j in range(qubits[1]+1, qubits[2]):
            XXX = kron(XXX, I_np)
        XXX = kron(XXX, X_np)
        for j in range(qubits[2]+1, qubits_number):
            XXX = kron(XXX, I_np)

        YYX = Y_np
        for j in range(qubits[0]):
            YYX = kron(I_np, YYX)
        for j in range(qubits[0]+1, qubits[1]):
            YYX = kron(YYX, I_np)
        YYX = kron(YYX, Y_np)
        for j in range(qubits[1]+1, qubits[2]):
            YYX = kron(YYX, I_np)
        YYX = kron(YYX, X_np)
        for j in range(qubits[2]+1, qubits_number):
            YYX = kron(YYX, I_np)

        ZZX = Z_np
        for j in range(qubits[0]):
            ZZX = kron(I_np, ZZX)
        for j in range(qubits[0]+1, qubits[1]):
            ZZX = kron(ZZX, I_np)
        ZZX = kron(ZZX, Z_np)
        for j in range(qubits[1]+1, qubits[2]):
            ZZX = kron(ZZX, I_np)
        ZZX = kron(ZZX, X_np)
        for j in range(qubits[2]+1, qubits_number):
            ZZX = kron(ZZX, I_np)

        IIX = I_np
        for j in range(qubits[2]-1):
            IIX = kron(I_np, IIX)
        IIX = kron(IIX, X_np)
        for j in range(qubits[2]+1, qubits_number):
            IIX = kron(IIX, I_np)

        su4_1 = unitary_matrix(operator_name='su4', qubits_number=qubits_number, qubits=qubits[:2], parameters=parameters[:15], daggered=daggered, derivatives=derivatives)
        su2_1 = unitary_matrix(operator_name='su2', qubits_number=qubits_number, qubits=[qubits[2]], parameters=parameters[15:18], daggered=daggered, derivatives=derivatives)

        op_1 = expm(-1j*(parameters[18]*XXZ + parameters[19]*YYZ + parameters[20]*ZZZ))

        su4_2 = unitary_matrix(operator_name='su4', qubits_number=qubits_number, qubits=qubits[:2], parameters=parameters[21:36], daggered=daggered, derivatives=derivatives)
        su2_2 = unitary_matrix(operator_name='su2', qubits_number=qubits_number, qubits=[qubits[2]], parameters=parameters[36:39], daggered=daggered, derivatives=derivatives)

        op_2 = expm(-1j*(parameters[39]*XXX + parameters[40]*YYX + parameters[41]*ZZX + parameters[42]*IIX))

        su4_3 = unitary_matrix(operator_name='su4', qubits_number=qubits_number, qubits=qubits[:2], parameters=parameters[43:58])
        su2_3 = unitary_matrix(operator_name='su2', qubits_number=qubits_number, qubits=[qubits[2]], parameters=parameters[58:61])

        op_3 = expm(-1j*(parameters[61]*XXZ + parameters[62]*YYZ + parameters[63]*ZZZ))

        su4_4 = unitary_matrix(operator_name='su4', qubits_number=qubits_number, qubits=qubits[:2], parameters=parameters[64:79])
        su2_4 = unitary_matrix(operator_name='su2', qubits_number=qubits_number, qubits=[qubits[2]], parameters=parameters[79:82])

        return dot_chain([su4_1, su2_1, op_1, su4_2, su2_2, op_2, su4_3, su2_3, op_3, su4_4, su2_4])

  ###############
  # other gates #
  ###############

    if operator_name == 'rank1':
              
        d_0 = derivatives.count(0)
        d_1 = derivatives.count(1)
        d_2 = derivatives.count(2)
        d_3 = derivatives.count(3)
        d_4 = derivatives.count(4)
        d_5 = derivatives.count(5)

        op_1 = expm(-1.j * (parameters[0] + d_0*pi/2) * X_np)
        op_2 = expm(-1.j * (parameters[1] + d_1*pi/2) * Y_np)
        op_3 = expm(-1.j * (parameters[2] + d_2*pi/2) * X_np)
        u3_1 = op_3.dot(op_2).dot(op_1)
        for j in range(qubits[0]):
            u3_1 = kron(I_np, u3_1)
        for j in range(qubits[0]+1, qubits_number):
            u3_1 = kron(u3_1, I_np)
            
        op_4 = expm(-1.j * (parameters[3] + d_3*pi/2) * X_np)
        op_5 = expm(-1.j * (parameters[4] + d_4*pi/2) * Y_np)
        op_6 = expm(-1.j * (parameters[5] + d_5*pi/2) * X_np)
        u3_2 = op_6.dot(op_5).dot(op_4)
        for j in range(qubits[1]):
            u3_2 = kron(I_np, u3_2)
        for j in range(qubits[1]+1, qubits_number):
            u3_2 = kron(u3_2, I_np)
            
        operator = u3_2.dot(u3_1)
        
        if daggered == False:
            return operator
        if daggered == True:
            return operator.conjugate().transpose()        
        
        ### old version ###
#         op_1 = expm(-1.j*parameters[0]*Z_np)
#         for i in range(derivatives.count(0)):
#             op_1 = op_1.dot(-1.j*Z_np)
#         op_2 = expm(-1.j*parameters[1]*X_np)
#         for i in range(derivatives.count(1)):
#             op_2 = op_2.dot(-1.j*X_np)
#         op_3 = expm(-1.j*parameters[2]*Z_np)
#         for i in range(derivatives.count(2)):
#             op_3 = op_3.dot(-1.j*Z_np)
#         u_3_1 = dot_chain([op_3, op_2, op_1])
#         for j in range(qubits[0]):
#             u_3_1 = kron(I_np, u_3_1)
#         for j in range(qubits[0]+1, qubits_number):
#             u_3_1 = kron(u_3_1, I_np)

#         op_1 = expm(-1.j*parameters[3]*Z_np)
#         for i in range(derivatives.count(3)):
#             op_1 = op_1.dot(-1.j*Z_np)
#         op_2 = expm(-1.j*parameters[4]*X_np)
#         for i in range(derivatives.count(4)):
#             op_2 = op_2.dot(-1.j*X_np)
#         op_3 = expm(-1.j*parameters[5]*Z_np)
#         for i in range(derivatives.count(5)):
#             op_3 = op_3.dot(-1.j*Z_np)
#         u_3_2 = dot_chain([op_3, op_2, op_1])
#         for j in range(qubits[1]):
#             u_3_2 = kron(I_np, u_3_2)
#         for j in range(qubits[1]+1, qubits_number):
#             u_3_2 = kron(u_3_2, I_np)

#         operator = u_3_2.dot(u_3_1)

#         if daggered == False:
#             return operator
#         if daggered == True:
#             return operator.conjugate().transpose()
    
    if operator_name == 'rank2':
        
        d_0 = derivatives.count(0)
        d_1 = derivatives.count(1)
        d_2 = derivatives.count(2)
        d_3 = derivatives.count(3)
        d_4 = derivatives.count(4)
        d_5 = derivatives.count(5)
        d_6 = derivatives.count(6)
        
        r_y = expm(-1.j * (parameters[0] + d_0*pi/2) * Y_np)
        for j in range(qubits[0]):
            r_y = kron(I_np, r_y)
        for j in range(qubits[0]+1, qubits_number):
            r_y = kron(r_y, I_np)

        cx_1 = P_0
        for i in range(qubits[0]):
            cx_1 = kron(I_np, cx_1)
        for i in range(qubits[0]+1, qubits_number):
            cx_1 = kron(cx_1, I_np)
        if qubits[0] < qubits[1]:
            op_1 = P_1
            op_2 = X_np
            q_1 = qubits[0]
            q_2 = qubits[1]
        if qubits[0] > qubits[1]:
            op_1 = X_np
            op_2 = P_1
            q_1 = qubits[1]
            q_2 = qubits[0]
        cx_2 = op_1
        for i in range(q_1):
            cx_2 = kron(I_np, cx_2)
        for i in range(q_1+1, q_2):
            cx_2 = kron(cx_2, I_np)
        cx_2 = kron(cx_2, op_2)
        for i in range(q_2+1, qubits_number):
            cx_2 = kron(cx_2, I_np)
        cx = cx_1 + cx_2
        
        op_1 = expm(-1.j * (parameters[1] + d_1*pi/2) * X_np)
        op_2 = expm(-1.j * (parameters[2] + d_2*pi/2) * Y_np)
        op_3 = expm(-1.j * (parameters[3] + d_3*pi/2) * X_np)
        u3_1 = op_3.dot(op_2).dot(op_1)
        for j in range(qubits[0]):
            u3_1 = kron(I_np, u3_1)
        for j in range(qubits[0]+1, qubits_number):
            u3_1 = kron(u3_1, I_np)
        
        op_4 = expm(-1.j * (parameters[4] + d_4*pi/2) * X_np)
        op_5 = expm(-1.j * (parameters[5] + d_5*pi/2) * Y_np)
        op_6 = expm(-1.j * (parameters[6] + d_6*pi/2) * X_np)
        u3_2 = op_6.dot(op_5).dot(op_4)
        for j in range(qubits[1]):
            u3_2 = kron(I_np, u3_2)
        for j in range(qubits[1]+1, qubits_number):
            u3_2 = kron(u3_2, I_np)
            
        operator = u3_2.dot(u3_1).dot(cx).dot(r_y)

        if daggered == False:
            return operator
        if daggered == True:
            return operator.conjugate().transpose()  
        
        ###### old version
#         r_y = expm(-1.j*parameters[0]*Y_np)
#         for i in range(derivatives.count(0)):
#             r_y = r_y.dot(-1.j*Y_np)
#         for j in range(qubits[0]):
#             r_y = kron(I_np, r_y)
#         for j in range(qubits[0]+1, qubits_number):
#             r_y = kron(r_y, I_np)

#         cx_1 = P_0
#         for i in range(qubits[0]):
#             cx_1 = kron(I_np, cx_1)
#         for i in range(qubits[0]+1, qubits_number):
#             cx_1 = kron(cx_1, I_np)
#         if qubits[0] < qubits[1]:
#             op_1 = P_1
#             op_2 = X_np
#             q_1 = qubits[0]
#             q_2 = qubits[1]
#         if qubits[0] > qubits[1]:
#             op_1 = X_np
#             op_2 = P_1
#             q_1 = qubits[1]
#             q_2 = qubits[0]
#         cx_2 = op_1
#         for i in range(q_1):
#             cx_2 = kron(I_np, cx_2)
#         for i in range(q_1+1, q_2):
#             cx_2 = kron(cx_2, I_np)
#         cx_2 = kron(cx_2, op_2)
#         for i in range(q_2+1, qubits_number):
#             cx_2 = kron(cx_2, I_np)
#         cx = cx_1 + cx_2

#         op_1 = expm(-1.j*parameters[1]*X_np)
#         for i in range(derivatives.count(1)):
#             op_1 = op_1.dot(-1.j*X_np)
#         op_2 = expm(-1.j*parameters[2]*Z_np)
#         for i in range(derivatives.count(2)):
#             op_2 = op_2.dot(-1.j*Z_np)
#         op_3 = expm(-1.j*parameters[3]*X_np)
#         for i in range(derivatives.count(3)):
#             op_3 = op_3.dot(-1.j*X_np)
#         u3_1 = dot_chain([op_3, op_2, op_1])
#         for j in range(qubits[0]):
#             u3_1 = kron(I_np, u3_1)
#         for j in range(qubits[0]+1, qubits_number):
#             u3_1 = kron(u3_1, I_np)

#         op_1 = expm(-1.j*parameters[4]*X_np)
#         for i in range(derivatives.count(4)):
#             op_1 = op_1.dot(-1.j*X_np)
#         op_2 = expm(-1.j*parameters[5]*Z_np)
#         for i in range(derivatives.count(5)):
#             op_2 = op_2.dot(-1.j*Z_np)
#         op_3 = expm(-1.j*parameters[6]*X_np)
#         for i in range(derivatives.count(6)):
#             op_3 = op_3.dot(-1.j*X_np)
#         u3_2 = dot_chain([op_3, op_2, op_1])
#         for j in range(qubits[1]):
#             u3_2 = kron(I_np, u3_2)
#         for j in range(qubits[1]+1, qubits_number):
#             u3_2 = kron(u3_2, I_np)

#         operator = u3_2.dot(u3_1).dot(cx).dot(r_y)

#         if daggered == False:
#             return operator
#         if daggered == True:
#             return operator.conjugate().transpose()
        
############################
        
    if operator_name == 'ising':

        d_0 = derivatives.count(0)
        d_1 = derivatives.count(1)
        d_2 = derivatives.count(2)
        d_3 = derivatives.count(3)
        d_4 = derivatives.count(4)          
            
        term_0 = expm(-1j * (parameters[0] + d_0*pi/2) * X_np) 
        for j in range(qubits[0]):
            term_0 = kron(I_np, term_0)
        for j in range(qubits[0]+1, qubits_number):
            term_0 = kron(term_0, I_np)     
            
#         term_0 = X_np
#         for j in range(qubits[0]):
#             term_0 = kron(I_np, term_0)
#         for j in range(qubits[0]+1, qubits_number):
#             term_0 = kron(term_0, I_np)
#         term_0 = expm(-1j * (parameters[0] + d_0*pi/2) * term_0)
    
        term_1 = expm(-1j * (parameters[1] + d_1*pi/2) * X_np)
        for j in range(qubits[1]):
            term_1 = kron(I_np, term_1)
        for j in range(qubits[1]+1, qubits_number):
            term_1 = kron(term_1, I_np)   
    
#         term_1 = X_np
#         for j in range(qubits[1]):
#             term_1 = kron(I_np, term_1)
#         for j in range(qubits[1]+1, qubits_number):
#             term_1 = kron(term_1, I_np)
#         term_1 = expm(-1j * (parameters[1] + d_1*pi/2) * term_1)

        term_2 = Z_np
        for j in range(qubits[0]):
            term_2 = kron(I_np, term_2)
        for j in range(qubits[0]+1, qubits[1]):
            term_2 = kron(term_2, I_np)
        term_2 = kron(term_2, Z_np)
        for j in range(qubits[1]+1, qubits_number):
            term_2 = kron(term_2, I_np)
        der_op = term_2
        term_2 = expm(-1j * (parameters[2] + d_2*pi/2) * term_2)

        term_3 = expm(-1j * (parameters[3] + d_3*pi/2) * Z_np)
        for j in range(qubits[0]):
            term_3 = kron(I_np, term_3)
        for j in range(qubits[0]+1, qubits_number):
            term_3 = kron(term_3, I_np)
        
#         term_3 = Z_np
#         for j in range(qubits[0]):
#             term_3 = kron(I_np, term_3)
#         for j in range(qubits[0]+1, qubits_number):
#             term_3 = kron(term_3, I_np)
#         term_3 = expm(-1j * (parameters[3] + d_3*pi/2) * term_3)
            
        term_4 = expm(-1j * (parameters[4] + d_4*pi/2) * Z_np) 
        for j in range(qubits[1]):
            term_4 = kron(I_np, term_4)
        for j in range(qubits[1]+1, qubits_number):
            term_4 = kron(term_4, I_np)
    
#         term_4 = Z_np
#         for j in range(qubits[1]):
#             term_4 = kron(I_np, term_4)
#         for j in range(qubits[1]+1, qubits_number):
#             term_4 = kron(term_4, I_np)
#         term_4 = expm(-1j * (parameters[4] + d_4*pi/2) * term_4)

        operator = term_4.dot(term_3).dot(term_2).dot(term_1).dot(term_0)

        if daggered == False:
            return operator
        if daggered == True:
            return operator.conjugate().transpose()


#         term_1 = X_np
#         for j in range(qubits[0]):
#             term_1 = kron(I_np, term_1)
#         for j in range(qubits[0]+1, qubits_number):
#             term_1 = kron(term_1, I_np)
#         der_op = term_1
#         term_1 = expm(-1j*parameters[0]*term_1)
#         for i in range(derivatives.count(0)):
#             term_1 = term_1.dot(-1.j*der_op)
            
#         term_2 = X_np
#         for j in range(qubits[1]):
#             term_2 = kron(I_np, term_2)
#         for j in range(qubits[1]+1, qubits_number):
#             term_2 = kron(term_2, I_np)
#         der_op = term_2
#         term_2 = expm(-1j*parameters[1]*term_2)
#         for i in range(derivatives.count(1)):
#             term_2 = term_2.dot(-1.j*der_op)
        
#         term_3 = Z_np
#         for j in range(qubits[0]):
#             term_3 = kron(I_np, term_3)
#         for j in range(qubits[0]+1, qubits[1]):
#             term_3 = kron(term_3, I_np)
#         term_3 = kron(term_3, Z_np)
#         for j in range(qubits[1]+1, qubits_number):
#             term_3 = kron(term_3, I_np)
#         der_op = term_3
#         term_3 = expm(-1j*parameters[2]*term_3)
#         for i in range(derivatives.count(2)):
#             term_3 = term_3.dot(-1.j*der_op)
        
#         term_4 = Z_np
#         for j in range(qubits[0]):
#             term_4 = kron(I_np, term_4)
#         for j in range(qubits[0]+1, qubits_number):
#             term_4 = kron(term_4, I_np)
#         der_op = term_4
#         term_4 = expm(-1j*parameters[3]*term_4)
#         for i in range(derivatives.count(3)):
#             term_4 = term_4.dot(-1.j*der_op)
            
#         term_5 = Z_np
#         for j in range(qubits[1]):
#             term_5 = kron(I_np, term_5)
#         for j in range(qubits[1]+1, qubits_number):
#             term_5 = kron(term_5, I_np)
#         der_op = term_5
#         term_5 = expm(-1j*parameters[4]*term_5)
#         for i in range(derivatives.count(4)):
#             term_5 = term_5.dot(-1.j*der_op)
        
#         operator = dot_chain([term_5, term_4, term_3, term_2, term_1])
        
#         if daggered == False:
#             return operator
#         if daggered == True:
#             return operator.conjugate().transpose()

###################
# tensor networks #
###################


def heas_ansatz(qubits_number, layers_number, target_qubits=None):

    heas_ansatz = []
    
    if target_qubits is None:
        target_qubits = [i for i in range(qubits_number)]
    
    target_qubits_number = len(target_qubits)
    
    block_number = 0
    for k in range(layers_number):

        for i in range(target_qubits_number):
            heas_ansatz.append({'layer_number': k,
                                'block_number': block_number,
                                'operator': 'r_x',
                                'qubits': [target_qubits[i]],
                                'parameters': [random.uniform(0*np.pi, 2*np.pi)],
                                'daggered': False,
                                'derivatives': []})
            block_number += 1
            
            heas_ansatz.append({'layer_number': k,
                                'block_number': block_number,
                                'operator': 'r_y',
                                'qubits': [target_qubits[i]],
                                'parameters': [random.uniform(0*np.pi, 2*np.pi)],
                                'daggered': False,
                                'derivatives': []})
            block_number += 1      
            
            heas_ansatz.append({'layer_number': k,
                                'block_number': block_number,
                                'operator': 'r_x',
                                'qubits': [target_qubits[i]],
                                'parameters': [random.uniform(0*np.pi, 2*np.pi)],
                                'daggered': False,
                                'derivatives': []})
            block_number += 1  
            
            

        for i in range(target_qubits_number - 1):
            heas_ansatz.append({'layer_number': k,
                                'block_number': block_number,
                                'operator': 'cx',
                                'qubits': [target_qubits[i], target_qubits[i+1]],
                                'parameters': [],
                                'daggered': False,
                                'derivatives': []})
            block_number += 1

        if target_qubits_number > 2:
            heas_ansatz.append({'layer_number': k,
                                'block_number': block_number,
                                'operator': 'cx',
                                'qubits': [target_qubits[-1], target_qubits[0]],
                                'parameters': [],
                                'daggered': False,
                                'derivatives': []})
            block_number += 1

    return heas_ansatz


def create_checkerboard_tensor_network(qubits_number, layers_number, operator_name, target_qubits=None):

    if operator_name == 'rank2':
        parameters_number = 7
    elif operator_name == 'rank1':
        parameters_number = 6
    elif operator_name == 'ising':
        parameters_number = 5
    elif operator_name == 'su2':
        parameters_number = 3
    elif operator_name == 'r_x' or 'r_y' or 'r_z':
        parameters_number = 1
    elif operator_name == 'su4':
        parameters_number = 15
        
    checkerboard = []

    if target_qubits is None:
        target_qubits = [i for i in range(qubits_number)]
    
    target_qubits_number = len(target_qubits)
    
    block_number = 0
    for k in range(layers_number):

        if k % 2 == 0:
            for i in range(int(target_qubits_number/2)):
                block = {'layer_number': k,
                         'block_number': block_number,
                         'operator': operator_name,
                         'qubits': [target_qubits[2*i], target_qubits[2*i+1]],
                         'parameters': [random.uniform(0, 2*np.pi) for i in range(parameters_number)],
                         'daggered': False,
                         'derivatives': []}
                checkerboard.append(block)#
                block_number += 1

        if k % 2 == 1:
            for i in range(int(target_qubits_number/2-1)):
                block = {'layer_number': k,
                         'block_number': block_number,
                         'operator': operator_name,
                         'qubits': [target_qubits[2*i+1], target_qubits[2*i+2]],
                         'parameters': [random.uniform(0, 2*np.pi) for i in range(parameters_number)],
                         'daggered': False,
                         'derivatives': []}
                checkerboard.append(block)
                block_number += 1

            block = {'layer_number':k,
                     'block_number': block_number,
                     'operator': operator_name,
                     'qubits': [target_qubits[0], target_qubits[-1]],
                     'parameters': [random.uniform(0, 2*np.pi) for i in range(parameters_number)],
                     'daggered': False,
                     'derivatives': []}
            checkerboard.append(block)
            block_number += 1

    return checkerboard


def hardware_efficient_ansatz_tensor_network(qubits_number, layers_number, target_qubits=None):

    hardware_efficient_ansatz = []
    
    if target_qubits is None:
        target_qubits = [i for i in range(qubits_number)]
    
    target_qubits_number = len(target_qubits)
    
    block_number = 0
    for k in range(layers_number):

        for i in range(target_qubits_number):
#             hardware_efficient_ansatz.append({'layer_number': k,
#                                               'block_number': block_number,
#                                               'operator': 'su2',
#                                               'qubits': [i],
#                                               'parameters': [random.uniform(-2*np.pi, 2*np.pi) for i in range(3)],
#                                               'daggered': False,
#                                               'derivatives': []})
#             block_number += 1
            
            hardware_efficient_ansatz.append({'layer_number': k,
                                              'block_number': block_number,
                                              'operator': 'r_x',
                                              'qubits': [target_qubits[i]],
                                              'parameters': [random.uniform(0*np.pi, 2*np.pi)],
                                              'daggered': False,
                                              'derivatives': []})
            block_number += 1
            
            hardware_efficient_ansatz.append({'layer_number': k,
                                              'block_number': block_number,
                                              'operator': 'r_y',
                                              'qubits': [target_qubits[i]],
                                              'parameters': [random.uniform(0*np.pi, 2*np.pi)],
                                              'daggered': False,
                                              'derivatives': []})
            block_number += 1      
            
            hardware_efficient_ansatz.append({'layer_number': k,
                                              'block_number': block_number,
                                              'operator': 'r_x',
                                              'qubits': [target_qubits[i]],
                                              'parameters': [random.uniform(0*np.pi, 2*np.pi)],
                                              'daggered': False,
                                              'derivatives': []})
            block_number += 1  
            
            

        for i in range(target_qubits_number - 1):
            hardware_efficient_ansatz.append({'layer_number': k,
                                              'block_number': block_number,
                                              'operator': 'cr_y',
                                              'qubits': [target_qubits[i], target_qubits[i+1]],
                                              'parameters': [random.uniform(0*np.pi, 2*np.pi)],
                                              'daggered': False,
                                              'derivatives': []})
            block_number += 1

        if target_qubits_number > 2:
            hardware_efficient_ansatz.append({'layer_number': k,
                                              'block_number': block_number,
                                              'operator': 'cr_y',
                                              'qubits': [target_qubits[-1], target_qubits[0]],
                                              'parameters': [random.uniform(0*np.pi, 2*np.pi)],
                                              'daggered': False,
                                              'derivatives': []})
            block_number += 1

    return hardware_efficient_ansatz



def create_single_blocks(qubits_number, operator_name):

    if operator_name == 'r_x' or 'r_y' or 'r_z':
        parameters_number = 1
    if operator_name == 'su2':
        parameters_number = 3

    single_blocks = []

    for k in range(qubits_number):
        single_blocks.append({'layer_number': 0,
                              'block_number': k,
                              'operator': operator_name, # unitary operation for the block
                              'qubits': [k], # list of the qubits which the operator acts on
                              'parameters': [random.uniform(0, 2.*np.pi) for i in range(parameters_number)], # some initial parameters
                              'daggered': False,
                              'derivatives': []})

    return single_blocks


def create_mps(qubits_number, operator_name):

    if operator_name == 'r_x' or 'r_y' or 'r_z':
        parameters_number = 1
    if operator_name == 'su2':
        parameters_number = 3
    if operator_name == 'alexey':
        parameters_number = 5
        gates_length = 2
    if operator_name == 'rank1':
        parameters_number = 6
        gates_length = 2
    if operator_name == 'rank2':
        parameters_number = 7
        gates_length = 2
    if operator_name == 'su4':
        parameters_number = 15
        gates_length = 2
    if operator_name == 'su8':
        parameters_number = 82
        gates_length = 3

    mps = []

    k = 0
    while(k+gates_length <= qubits_number):
        mps.append({'layer_number': 0,
                    'block_number': k,
                    'operator': operator_name,
                    'qubits': np.arange(k, k+gates_length, 1),
                    'parameters': [random.uniform(0, 2.*np.pi) for i in range(parameters_number)],
                    'daggered': False,
                    'derivatives': []})

        k += 1
    return mps


def create_tree(qubits_number, layers_number, operator_name):

    if operator_name == 'r_x' or 'r_y' or 'r_z':
        parameters_number = 1
    if operator_name == 'u_3':
        parameters_number = 3
    if operator_name == 'alexey':
        parameters_number = 5
    if operator_name == 'rank1':
        parameters_number = 6
    if operator_name == 'rank2':
        parameters_number = 7

    tree = []

    tree.append({'layer_number': 0,
                 'block_number': 0,
                 'operator': operator_name,
                 'qubits': [int(qubits_number/2)-1, int(qubits_number/2)],
                 'parameters': [random.uniform(0, 2*np.pi) for i in range(parameters_number)],
                 'daggered': False,
                 'derivatives': []})

    block_number = 1
    for k in range(1, layers_number):
        block = {'layer_number': k,
                 'block_number': block_number,
                 'operator': operator_name,
                 'qubits': [int(qubits_number/2)-k-1, int(qubits_number/2)-k],
                 'parameters': [random.uniform(0, 2*np.pi) for i in range(parameters_number)],
                 'daggered': False,
                 'derivatives': []}
        tree.append(block)
        block_number += 1

        block = {'layer_number': k,
                 'block_number': block_number,
                 'operator': operator_name,
                 'qubits': [int(qubits_number/2)+k-1, int(qubits_number/2)+k],
                 'parameters': [random.uniform(0, 2*np.pi) for i in range(parameters_number)],
                 'daggered': False,
                 'derivatives': []}
        tree.append(block)
        block_number += 1

    return tree


# rewrite this function to be able to deal with empty tensor networks
def sum_tensor_networks(*tn_args):

    # to avoid modification of the initial tensor networks
    tensor_networks = copy.deepcopy(tn_args)

    sum_tn = tensor_networks[0]

    tn_counter = 1

    for tn in tensor_networks[1:]:

        maximum_layers_number = sum_tn[-1]['layer_number'] + tn_counter

        # update layer numbers
        for block in tn:
            block['layer_number'] += maximum_layers_number

        # concatenate tensor networks
        sum_tn = sum_tn + tn
        tn_counter += 1

    #update block numbers
    block_counter = 0
    for block in sum_tn:
        block['block_number'] = block_counter
        block_counter +=1

    return sum_tn

def daggerize_tensor_network(tensor_network):

    daggered_tensor_network = copy.deepcopy(tensor_network[::-1])
    for block in daggered_tensor_network:
        block['daggered'] = True

    blocks_number = len(daggered_tensor_network)

    layer_numbers = []
    for k in range(blocks_number):
        daggered_tensor_network[k]['block_number'] = k
        layer_numbers.append(daggered_tensor_network[k]['layer_number'])
    for k in range(blocks_number):
        daggered_tensor_network[k]['layer_number'] = layer_numbers[blocks_number-k-1]

    return daggered_tensor_network


def update_tensor_network(tensor_network, new_parameters):

    parameters_updated = 0

    for block in tensor_network:
        parameters_number = len(block['parameters'])
        block['parameters'] = new_parameters[parameters_updated : parameters_updated+parameters_number]
        parameters_updated += parameters_number

    return None


def tensor_network_parameters_number(tensor_network):

    parameters_number = 0

    for block in tensor_network:
        parameters_number += len(block['parameters'])

    return parameters_number


def extract_parameters_from_tensor_network(tensor_network):
    
    parameters = []
    
    for block in tensor_network:
        parameters = parameters + list(block['parameters'])
    
    return parameters


def tensor_network_to_matrix(qubits_number, tensor_network):
    """
    Input:
        number of qubits,
        tensor network
    Return:
        NumPy matrix representation of a tensor network
    """
    matrix = I_np
    for i in range(qubits_number-1):
        matrix = kron(matrix, I_np)

    for block in tensor_network:
        matrix = unitary_matrix(operator_name=block['operator'],
                                qubits_number=qubits_number,
                                qubits=block['qubits'],
                                parameters=block['parameters'],
                                daggered=block['daggered'],
                                derivatives=block['derivatives']).dot(matrix)

    return matrix


def tensor_network_to_state_vector(qubits_number, tensor_network):
    """
    Input:
        number of qubits,
        tensor network
    Return:
        NumPy vector U.|0> with U being the tensor network matrix
    """
    
    matrix = tensor_network_to_matrix(qubits_number, tensor_network)
#     zero_state = kron_chain([s0 for i in range(qubits_number)])
    zero_state = [1] + [0 for i in range(2**qubits_number - 1)]
    
    return matrix.dot(zero_state)


def tensor_network_to_qiskit_circuit(qubits_number, tensor_network, quantum_register, classical_register, beo=True):
    
    circuit = QuantumCircuit(quantum_register, classical_register)
    
    layers_passed = 0    
    for block in tensor_network:

        if beo == True:
            block['qubits'] = [(qubits_number - 1 - i) for i in block['qubits']]
        
        if block['operator'] == 'r_x':
            circuit.rx(2*block['parameters'][0] + pi*block['derivatives'].count(0), block['qubits'][0])
        elif block['operator'] == 'r_y':
            circuit.ry(2*block['parameters'][0] + pi*block['derivatives'].count(0), block['qubits'][0])
        elif block['operator'] == 'r_z':
            circuit.rz(2*block['parameters'][0] + pi*block['derivatives'].count(0), block['qubits'][0])
        elif block['operator'] == 'cx':
            print(block['operator'], block['derivatives'].count(0))
            circuit.cx(block['qubits'][0], block['qubits'][1])
        elif block['operator'] == 'cz':
            circuit.cz(block['qubits'][0], block['qubits'][1])            
        elif block['operator'] == 'cr_y':
            circuit.cu3(2*block['parameters'][0] + pi*block['derivatives'].count(0), 0, 0, block['qubits'][0], block['qubits'][1])
        elif block['operator'] == 'su2':
            circuit.rx(2*block['parameters'][0] + pi*block['derivatives'].count(0), block['qubits'][0])
            circuit.rz(2*block['parameters'][1] + pi*block['derivatives'].count(1), block['qubits'][0])
            circuit.rx(2*block['parameters'][2] + pi*block['derivatives'].count(2), block['qubits'][0])
        elif block['operator'] == 'ising':
            circuit.rx(2*block['parameters'][0] + pi*block['derivatives'].count(0), block['qubits'][0])
            circuit.rx(2*block['parameters'][1] + pi*block['derivatives'].count(1), block['qubits'][1])
            circuit.rzz(2*block['parameters'][2] + pi*block['derivatives'].count(2), block['qubits'][0], block['qubits'][1])
            circuit.rz(2*block['parameters'][3] + pi*block['derivatives'].count(3), block['qubits'][0])
            circuit.rz(2*block['parameters'][4] + pi*block['derivatives'].count(4), block['qubits'][1])
        elif block['operator'] == 'rank1':
            circuit.rx(2*block['parameters'][0] + pi*block['derivatives'].count(0), block['qubits'][0])
            circuit.ry(2*block['parameters'][1] + pi*block['derivatives'].count(1), block['qubits'][0])
            circuit.rx(2*block['parameters'][2] + pi*block['derivatives'].count(2), block['qubits'][0])
            circuit.rx(2*block['parameters'][3] + pi*block['derivatives'].count(3), block['qubits'][1])
            circuit.ry(2*block['parameters'][4] + pi*block['derivatives'].count(4), block['qubits'][1])
            circuit.rx(2*block['parameters'][5] + pi*block['derivatives'].count(5), block['qubits'][1])
        elif block['operator'] == 'rank2':
            circuit.ry(2*block['parameters'][0] + pi*block['derivatives'].count(0), block['qubits'][0])
            circuit.cx(block['qubits'][0], block['qubits'][1])
            circuit.rx(2*block['parameters'][1] + pi*block['derivatives'].count(1), block['qubits'][0])
            circuit.ry(2*block['parameters'][2] + pi*block['derivatives'].count(2), block['qubits'][0])
            circuit.rx(2*block['parameters'][3] + pi*block['derivatives'].count(3), block['qubits'][0])
            circuit.rx(2*block['parameters'][4] + pi*block['derivatives'].count(4), block['qubits'][1])
            circuit.ry(2*block['parameters'][5] + pi*block['derivatives'].count(5), block['qubits'][1])
            circuit.rx(2*block['parameters'][6] + pi*block['derivatives'].count(6), block['qubits'][1])

        if beo == True:
            block['qubits'] = [(qubits_number - 1 - i) for i in block['qubits']]
            
        # for testing purposes
#         if block['layer_number'] != layers_passed:
#             layers_passed = block['layer_number']
#             circuit.barrier() 
                
    return circuit


# def tensor_network_to_projectq_state(qubits_number, state_tensor_network, quantum_state=None):
        
#     layers_passed = 0    
#     for block in state_tensor_network:
        
#         if block['operator'] == 'r_x':            
#             Rx(block['parameters'][0] + pi/2*block['derivatives'].count(0)) | quantum_state[block['qubits'][0]]
#         elif block['operator'] == 'r_y':
#             Ry(block['parameters'][0] + pi/2*block['derivatives'].count(0)) | quantum_state[block['qubits'][0]]
#         elif block['operator'] == 'r_z':
#             Rz(block['parameters'][0] + pi/2*block['derivatives'].count(0)) | quantum_state[block['qubits'][0]]
#         elif block['operator'] == 'cx':
#             C(X) | (quantum_state[block['qubits'][0]], quantum_state[block['qubits'][1]])
#         elif block['operator'] == 'cr_y':
#             C(Ry(block['parameters'][0] + pi*block['derivatives'].count(0))) | (quantum_state[block['qubits'][0]], quantum_state[block['qubits'][1]])
#         elif block['operator'] == 'su2':
#             Rx(block['parameters'][0] + pi/2*block['derivatives'].count(0)) | quantum_state[block['qubits'][0]]
#             Ry(block['parameters'][1] + pi/2*block['derivatives'].count(1)) | quantum_state[block['qubits'][0]]
#             Rx(block['parameters'][2] + pi/2*block['derivatives'].count(2)) | quantum_state[block['qubits'][0]]
#         elif block['operator'] == 'ising':
#             Rx(block['parameters'][0] + pi/2*block['derivatives'].count(0)) | quantum_state[block['qubits'][0]]
#             Rx(block['parameters'][1] + pi/2*block['derivatives'].count(1)) | quantum_state[block['qubits'][1]]
#             TimeEvolution((parameters[2] + pi/2*block['derivatives'].count(2)), QubitOperator("Z"+str(block['qubits'][0]) + " Z"+str(block['qubits'][1]))) | quantum_state 
#             Rz(block['parameters'][3] + pi/2*block['derivatives'].count(3)) | quantum_state[block['qubits'][0]]
#             Rz(block['parameters'][4] + pi/2*block['derivatives'].count(4)) | quantum_state[block['qubits'][1]]
#         elif block['operator'] == 'rank1':
#             Rx(block['parameters'][0] + pi/2*block['derivatives'].count(0)) | quantum_state[block['qubits'][0]]
#             Ry(block['parameters'][1] + pi/2*block['derivatives'].count(1)) | quantum_state[block['qubits'][0]]
#             Rx(block['parameters'][2] + pi/2*block['derivatives'].count(2)) | quantum_state[block['qubits'][0]]
#             Rx(block['parameters'][3] + pi/2*block['derivatives'].count(3)) | quantum_state[block['qubits'][1]]
#             Ry(block['parameters'][4] + pi/2*block['derivatives'].count(4)) | quantum_state[block['qubits'][1]]
#             Rx(block['parameters'][5] + pi/2*block['derivatives'].count(5)) | quantum_state[block['qubits'][1]]
#         elif block['operator'] == 'rank2':
#             Ry(block['parameters'][0] + pi/2*block['derivatives'].count(0)) | quantum_state[block['qubits'][0]]
#             C(X) | (quantum_state[block['qubits'][0]], quantum_state[block['qubits'][1]])
#             Rx(block['parameters'][1] + pi/2*block['derivatives'].count(1)) | quantum_state[block['qubits'][0]]
#             Ry(block['parameters'][2] + pi/2*block['derivatives'].count(2)) | quantum_state[block['qubits'][0]]
#             Rx(block['parameters'][3] + pi/2*block['derivatives'].count(3)) | quantum_state[block['qubits'][0]]
#             Rx(block['parameters'][4] + pi/2*block['derivatives'].count(4)) | quantum_state[block['qubits'][1]]
#             Ry(block['parameters'][5] + pi/2*block['derivatives'].count(5)) | quantum_state[block['qubits'][1]]
#             Rx(block['parameters'][6] + pi/2*block['derivatives'].count(6)) | quantum_state[block['qubits'][1]]
            
#     return None


def tensor_network_to_projectq_state(qubits_number, state_tensor_network, quantum_state=None, engine=None, beo=True):
        
    if quantum_state is None:
        quantum_state = engine.allocate_qureg(qubits_number)
    
    layers_passed = 0    
    for block in state_tensor_network:
        
        if beo == True:
            block['qubits'] = [(qubits_number - 1 - i) for i in block['qubits']]
        
        if block['operator'] == 'r_x':
            Rx(2*block['parameters'][0] + pi*block['derivatives'].count(0)) | quantum_state[block['qubits'][0]]
        elif block['operator'] == 'r_y':
            Ry(2*block['parameters'][0] + pi*block['derivatives'].count(0)) | quantum_state[block['qubits'][0]]
        elif block['operator'] == 'r_z':
            Rz(2*block['parameters'][0] + pi*block['derivatives'].count(0)) | quantum_state[block['qubits'][0]]
        elif block['operator'] == 'cx':
            C(X) | (quantum_state[block['qubits'][0]], quantum_state[block['qubits'][1]])
        elif block['operator'] == 'cr_y':
            C(Ry(block['parameters'][0] + pi*block['derivatives'].count(0))) | (quantum_state[block['qubits'][0]], quantum_state[block['qubits'][1]])
        elif block['operator'] == 'su2':
            Rx(block['parameters'][0] + pi/2*block['derivatives'].count(0)) | quantum_state[block['qubits'][0]]
            Ry(block['parameters'][1] + pi/2*block['derivatives'].count(1)) | quantum_state[block['qubits'][0]]
            Rx(block['parameters'][2] + pi/2*block['derivatives'].count(2)) | quantum_state[block['qubits'][0]]
        elif block['operator'] == 'ising':
            Rx(block['parameters'][0] + pi/2*block['derivatives'].count(0)) | quantum_state[block['qubits'][0]]
            Rx(block['parameters'][1] + pi/2*block['derivatives'].count(1)) | quantum_state[block['qubits'][1]]
            TimeEvolution((parameters[2] + pi/2*block['derivatives'].count(2)), QubitOperator("Z"+str(block['qubits'][0]) + " Z"+str(block['qubits'][1]))) | quantum_state 
            Rz(block['parameters'][3] + pi/2*block['derivatives'].count(3)) | quantum_state[block['qubits'][0]]
            Rz(block['parameters'][4] + pi/2*block['derivatives'].count(4)) | quantum_state[block['qubits'][1]]
        elif block['operator'] == 'rank1':
            Rx(block['parameters'][0] + pi/2*block['derivatives'].count(0)) | quantum_state[block['qubits'][0]]
            Ry(block['parameters'][1] + pi/2*block['derivatives'].count(1)) | quantum_state[block['qubits'][0]]
            Rx(block['parameters'][2] + pi/2*block['derivatives'].count(2)) | quantum_state[block['qubits'][0]]
            Rx(block['parameters'][3] + pi/2*block['derivatives'].count(3)) | quantum_state[block['qubits'][1]]
            Ry(block['parameters'][4] + pi/2*block['derivatives'].count(4)) | quantum_state[block['qubits'][1]]
            Rx(block['parameters'][5] + pi/2*block['derivatives'].count(5)) | quantum_state[block['qubits'][1]]
        elif block['operator'] == 'rank2':
            Ry(block['parameters'][0] + pi/2*block['derivatives'].count(0)) | quantum_state[block['qubits'][0]]
            C(X) | (quantum_state[block['qubits'][0]], quantum_state[block['qubits'][1]])
            Rx(block['parameters'][1] + pi/2*block['derivatives'].count(1)) | quantum_state[block['qubits'][0]]
            Ry(block['parameters'][2] + pi/2*block['derivatives'].count(2)) | quantum_state[block['qubits'][0]]
            Rx(block['parameters'][3] + pi/2*block['derivatives'].count(3)) | quantum_state[block['qubits'][0]]
            Rx(block['parameters'][4] + pi/2*block['derivatives'].count(4)) | quantum_state[block['qubits'][1]]
            Ry(block['parameters'][5] + pi/2*block['derivatives'].count(5)) | quantum_state[block['qubits'][1]]
            Rx(block['parameters'][6] + pi/2*block['derivatives'].count(6)) | quantum_state[block['qubits'][1]]
            
        if beo == True:
            block['qubits'] = [(qubits_number - 1 - i) for i in block['qubits']]

    if quantum_state is None:
        return None 
    else:
        return quantum_state



###################################################
# Expectation value / Matrix element of a unitary #
###################################################

# ProjectQ #

def real_imaginary_part_expectation_value(part, qubits_number, unitary_tensor_network, engine, initial_state_tensor_network=[]):
    '''
    performs the Hadamard test
    returns the real or imaginary part of the expectation value of a unitary operator in some state
    by default the initial state is |00...0>
    both the initial state and unitary operator are the tensor networks
    '''

    quantum_state = engine.allocate_qureg(qubits_number)

    # prepare the auxiliary qubit
    auxiliary_qubit = engine.allocate_qubit()

    H | auxiliary_qubit

    if part == 'Im':
        Sdag | auxiliary_qubit

    with Control(engine, auxiliary_qubit):

        # prepares the initial state U |00...0> if needed
        # U is the identity by default
        for block in initial_state_tensor_network:

            # generate the list of unitaries corresponing to the current block
            block_unitaries_list = unitary_operator(qubits_number = qubits_number,
                                                    operator_name = block['operator'],
                                                    qubits = block['qubits'],
                                                    parameters = block['parameters'],
                                                    daggered = block['daggered'],
                                                    derivatives = block['derivatives'])

            # apply all the unitaries to the state one by one
            for unitary in block_unitaries_list:
                unitary | quantum_state
        # send all the operations to the backend
        engine.flush()

        # apply the unitary; gives Q U |00...0>
        for block in unitary_tensor_network:

            block_unitaries_list = unitary_operator(qubits_number = qubits_number,
                                                    operator_name = block['operator'],
                                                    qubits = block['qubits'],
                                                    parameters = block['parameters'],
                                                    daggered = block['daggered'],
                                                    derivatives = block['derivatives'])

            for unitary in block_unitaries_list:
                unitary | quantum_state

        engine.flush()

    H | auxiliary_qubit

    engine.flush()

    # calculate the probability to measure the auxiliary qubit in the state |0>
    probability_0 = engine.backend.get_probability('0', auxiliary_qubit)

    Measure | auxiliary_qubit
    All(Measure) | quantum_state
    engine.flush()

    return 2*probability_0 - 1


def unitary_expectation_value(qubits_number, unitary_tensor_network, engine, initial_state_tensor_network=[]):

    real_part = real_imaginary_part_expectation_value('Re', qubits_number, unitary_tensor_network, engine, initial_state_tensor_network)

    imaginary_part = real_imaginary_part_expectation_value('Im', qubits_number, unitary_tensor_network, engine, initial_state_tensor_network)

    return real_part + 1j*imaginary_part

# NumPy #

# at the moment does not work for the empty tensor networks
def real_imaginary_part_expectation_value_numpy(part, qubits_number, unitary_tensor_network, engine, initial_state_tensor_network=[]):
    '''
    the same function but for NumPy
    '''

    #concatenated_tensor_network = sum_tensor_networks(initial_state_tensor_network, unitary_tensor_network)
    concatenated_tensor_network = sum_tensor_networks(unitary_tensor_network)
    matrix = tensor_network_to_matrix(qubits_number, concatenated_tensor_network)
    zero_state = kron_chain([s0 for i in range(qubits_number)])

    expected_value = dot_chain([zero_state.conjugate().transpose(), matrix, zero_state])

    if part == 'Re':
        return np.real(dot_chain([zero_state.conjugate().transpose(), matrix, zero_state]))
    if part == 'Im':
        return np.imag(dot_chain([zero_state.conjugate().transpose(), matrix, zero_state]))


def unitary_expectation_value_numpy(qubits_number, unitary_tensor_network, engine, initial_state_tensor_network=[]):

    real_part = real_imaginary_part_expectation_value_numpy('Re', qubits_number, unitary_tensor_network, engine, initial_state_tensor_network)

    imaginary_part = real_imaginary_part_expectation_value_numpy('Im', qubits_number, unitary_tensor_network, engine, initial_state_tensor_network)

    return (real_part + 1j*imaginary_part)[0][0]


###################################
# Matrix element of a Hamiltonian #
###################################


def ising_model(qubits_number, J, hx):
    '''
    returns a Hamiltonian represented as a dictionary
    example: {'ZZ': -1, 'XI': 0.5, 'IX': 0.5}
    '''

    ham = {}

    line = 'Z' + 'Z' + 'I' * (qubits_number - 2)

    for i in range(qubits_number):
        term = line[-i:] + line[:-i]
        ham[term] = J
    line = 'X' + 'I' * (qubits_number - 1)
    if hx != 0:
        for i in range(qubits_number):
            term = line[-i:] + line[:-i]
            ham[term] = hx

    return ham

def heisenberg_model(qubits_number, Jx, Jy, Jz, hz):

    ham = {}

    line = 'X' + 'X' + 'I' * (qubits_number - 2)
    for i in range(qubits_number):
        term = line[-i:] + line[:-i]
        ham[term] = Jx

    line = 'Y' + 'Y' + 'I' * (qubits_number - 2)
    for i in range(qubits_number):
        term = line[-i:] + line[:-i]
        ham[term] = Jy

    line = 'Z' + 'Z' + 'I' * (qubits_number - 2)
    for i in range(qubits_number):
        term = line[-i:] + line[:-i]
        ham[term] = Jz

    line = 'Z' + 'I' * (qubits_number - 1)
    if hz != 0:
        for i in range(qubits_number):
            term = line[-i:] + line[:-i]
            ham[term] = hz

    return ham


def insert_operator_in_hamiltonian(hamiltonian_dictionary, operator, position):
    
    hamiltonian_dictionary_new = {}
    
    for pauli_string, multiplier in hamiltonian_dictionary.items():
        
        pauli_string_new = pauli_string[:position] + operator + pauli_string[position:]

        hamiltonian_dictionary_new[pauli_string_new] = multiplier

    return hamiltonian_dictionary_new


def remove_operator_from_hamiltonian(hamiltonian_dictionary, position):
    
    hamiltonian_dictionary_new = {}
    
    for pauli_string, multiplier in hamiltonian_dictionary.items():
        
        pauli_string_new = pauli_string[:position] + pauli_string[position+1:]

        hamiltonian_dictionary_new[pauli_string_new] = multiplier

    return hamiltonian_dictionary_new


def hamiltonian_matrix_element_part(part, qubits_number, state_tensor_network, daggered_state_tensor_network, hamiltonian_dictionary, engine):
    '''
    takes a Hamltonian and two tensor networks
    returns the real or imaginary part of the matrix element
    <0|daggered_state_tensor_network . Hamiltonian . state_tensor_network|0>
    '''  
    
    expectation_value = 0
    for operator, multiplier in hamiltonian_dictionary.items():

        # prepare the main quantum state
        quantum_state = engine.allocate_qureg(qubits_number)

        # prepare the auxiliary qubit
        auxiliary_qubit = engine.allocate_qubit()

        H | auxiliary_qubit
        if part == 'Im':
            Sdag | auxiliary_qubit

        with Control(engine, auxiliary_qubit):

            # apply state tensor network; gives U.|00...0>
            for block in state_tensor_network:
                block_unitaries_list = unitary_operator(qubits_number = qubits_number,
                                                        operator_name = block['operator'],
                                                        qubits = block['qubits'],
                                                        parameters = block['parameters'],
                                                        daggered = block['daggered'],
                                                        derivatives = block['derivatives'])
                for unitary in block_unitaries_list:
                    unitary | quantum_state

            # apply a term of the Hamiltonian; gives term.U.|00...0>
            temp_operator = QubitOperator('')
            for k in range(qubits_number):
                if operator[k] != 'I':
                    temp_operator *= QubitOperator(operator[k]+str(k))
            temp_operator | quantum_state

            # apply daggered state tensor network; gives U^\dagger.term.U.|00...0>
            for block in daggered_state_tensor_network:
                block_unitaries_list = unitary_operator(qubits_number = qubits_number,
                                                        operator_name = block['operator'],
                                                        qubits = block['qubits'],
                                                        parameters = block['parameters'],
                                                        daggered = block['daggered'],
                                                        derivatives = block['derivatives'])
                for unitary in block_unitaries_list:
                    unitary | quantum_state

        H | auxiliary_qubit

        engine.flush()

        # calculate the probability to measure the auxiliary qubit in the state |0>
        probability_0 = engine.backend.get_probability('0', auxiliary_qubit)

        Measure | auxiliary_qubit
        All(Measure) | quantum_state

        expectation_value = expectation_value + multiplier*(2*probability_0 - 1)

    return expectation_value



def hamiltonian_matrix_element_part_projectq_cheaty(part, qubits_number, state_tensor_network, daggered_state_tensor_network, hamiltonian_matrix):

    simulator = Simulator()
    engine = MainEngine(backend=simulator)    
    quantum_state_daggerized = engine.allocate_qureg(qubits_number)
    for block in daggered_state_tensor_network:
#         block['qubits'] = [(qubits_number - 1 - i) for i in block['qubits']]
        block_unitaries_list = unitary_operator(qubits_number = qubits_number,
                                                operator_name = block['operator'],
                                                qubits = block['qubits'],
                                                parameters = block['parameters'],
                                                daggered = block['daggered'],
                                                derivatives = block['derivatives'])
        for unitary in block_unitaries_list:
            unitary | quantum_state_daggerized
#         block['qubits'] = [(qubits_number - 1 - i) for i in block['qubits']]
    engine.flush()
    state_vector_daggerized = np.array(simulator.cheat()[1])
    All(Measure) | quantum_state_daggerized
    del(quantum_state_daggerized) # hacks, probably    
    
    simulator = Simulator()
    engine = MainEngine(backend=simulator)
    quantum_state = engine.allocate_qureg(qubits_number)
    for block in state_tensor_network:
#         block['qubits'] = [(qubits_number - 1 - i) for i in block['qubits']]
        block_unitaries_list = unitary_operator(qubits_number = qubits_number,
                                                operator_name = block['operator'],
                                                qubits = block['qubits'],
                                                parameters = block['parameters'],
                                                daggered = block['daggered'],
                                                derivatives = block['derivatives'])
        for unitary in block_unitaries_list:
            unitary | quantum_state
#         block['qubits'] = [(qubits_number - 1 - i) for i in block['qubits']]
    engine.flush()
    state_vector = np.array(simulator.cheat()[1])
    All(Measure) | quantum_state
    del(quantum_state) # hacks, probably
    
    if part == 'Re':
        matrix_element = (state_vector_daggerized.dot(hamiltonian_matrix).dot(state_vector)).real
    elif part == 'Im':
        matrix_element = (state_vector_daggerized.dot(hamiltonian_matrix).dot(state_vector)).imag
    return matrix_element


def hamiltonian_matrix_element_part_numpy(part, qubits_number, state_tensor_network, daggered_state_tensor_network, hamiltonian):
    '''
    the same function but for NumPy
    '''
    if type(hamiltonian) is dict:
        hamiltonian = hamiltonian_dictionary_to_matrix(hamiltonian)

    state_vector = tensor_network_to_matrix(qubits_number, state_tensor_network).dot(kron_chain([s0 for i in range(qubits_number)]))
    daggered_state_vector = kron_chain([s0 for i in range(qubits_number)]).conjugate().transpose().dot(tensor_network_to_matrix(qubits_number, daggered_state_tensor_network))

    if part == 'Re':
        return np.real(dot_chain([daggered_state_vector, hamiltonian, state_vector]))[0][0]
    if part == 'Im':
        return np.imag(dot_chain([daggered_state_vector, hamiltonian, state_vector]))[0][0]


def hamiltonian_dictionary_to_operator(qubits_number, hamiltonian_dictionary):
    '''
    transforms a Hamiltonian from a dictionary to a ProjectQ object
    '''

    hamiltonian_operator = QubitOperator("")

    for operator, multiplier in hamiltonian_dictionary.items():

        temp_operator = QubitOperator("")
        for k in range(qubits_number):
            if operator[k] != 'I':
                temp_operator *= QubitOperator(operator[k]+str(k))
        temp_operator *= multiplier

        hamiltonian_operator += temp_operator

    return hamiltonian_operator - QubitOperator("")


def hamiltonian_dictionary_to_matrix(hamiltonian_dictionary):
    '''
    transforms a Hamiltonian from a dictionary to a NumPy matrix
    '''

    hamiltonian_m = 0

    for operator, multiplier in hamiltonian_dictionary.items():

        if operator[0] == 'I':
            hamiltonian_term = I_np
        elif operator[0] == 'X':
            hamiltonian_term = X_np
        elif operator[0] == 'Y':
            hamiltonian_term = Y_np
        elif operator[0] == 'Z':
            hamiltonian_term = Z_np

        for k in range(1, len(operator)):
            if operator[k] == 'I':
                hamiltonian_term = kron(hamiltonian_term, I_np)
            elif operator[k] == 'X':
                hamiltonian_term = kron(hamiltonian_term, X_np)
            elif operator[k] == 'Y':
                hamiltonian_term = kron(hamiltonian_term, Y_np)
            elif operator[k] == 'Z':
                hamiltonian_term = kron(hamiltonian_term, Z_np)

        hamiltonian_m = hamiltonian_m + multiplier*hamiltonian_term

    return hamiltonian_m

######################################
# Cost function gradient and Hessian #
######################################

## VQE ##

# ProjectQ #

def vqe_cost_function_gradient(qubits_number, state_tensor_network, hamiltonian_dictionary, engine):
    '''
    returns the gradient of the cost function of VQE as a NumPy vector
    '''

    daggered_state_tensor_network = daggerize_tensor_network(state_tensor_network)

    components = []

    for block in state_tensor_network:

        for parameter_number in range(len(block['parameters'])):

            block['derivatives'].append(parameter_number)

            components.append(hamiltonian_matrix_element_part('Re', qubits_number, state_tensor_network, daggered_state_tensor_network, hamiltonian_dictionary, engine))

            block['derivatives'].pop(-1)

    return 2*np.array(components)


def vqe_cost_function_gradient_projectq_cheaty(qubits_number, state_tensor_network, hamiltonian_matrix):

    daggered_state_tensor_network = daggerize_tensor_network(state_tensor_network)
    
    components = []

    for block in state_tensor_network:

        for parameter_number in range(len(block['parameters'])):
            
            block['derivatives'].append(parameter_number)
            
            components.append(hamiltonian_matrix_element_part_projectq_cheaty('Re', qubits_number, state_tensor_network, daggered_state_tensor_network, hamiltonian_matrix))

            block['derivatives'].pop(-1)

    return 2*np.array(components)


def perform_vqe_gradient_descent_projectq(qubits_number, hamiltonian_dictionary, state_tensor_network, engine, gamma=0.1, iterations_number=20):
    '''
    performs dummy gradient descent without termination conditions and with fixed step size \gamma
    '''

    daggered_state_tensor_network = daggerize_tensor_network(state_tensor_network)

    hamiltonian_expectation_values = [hamiltonian_matrix_element_part('Re', qubits_number, state_tensor_network, daggered_state_tensor_network, hamiltonian_dictionary, engine)]

    for iteration in range(iterations_number):

        gradient_current = vqe_cost_function_gradient(qubits_number, state_tensor_network, hamiltonian_dictionary, engine)

        parameters_set_current = []
        for block in state_tensor_network:
            parameters_set_current = np.concatenate((parameters_set_current, block['parameters']))
        parameters_set_next = parameters_set_current - gamma*gradient_current

        parameters_updated = 0
        for block in state_tensor_network:
            parameters_number = len(block['parameters'])
            block['parameters'] = parameters_set_next[parameters_updated : parameters_updated+parameters_number]
            parameters_updated += parameters_number

        daggered_state_tensor_network = daggerize_tensor_network(state_tensor_network)

        hamiltonian_expectation_values.append(hamiltonian_matrix_element_part('Re', qubits_number, state_tensor_network, daggered_state_tensor_network, hamiltonian_dictionary, engine))

    return hamiltonian_expectation_values


def perform_vqe_gradient_descent_projectq_cheaty(qubits_number, hamiltonian_dictionary, state_tensor_network, gamma=0.1, iterations_number=20):

    hamiltonian_matrix = hamiltonian_dictionary_to_matrix(hamiltonian_dictionary)
    
    daggered_state_tensor_network = daggerize_tensor_network(state_tensor_network)
    
    hamiltonian_expectation_values = [hamiltonian_matrix_element_part_projectq_cheaty('Re', qubits_number, state_tensor_network, daggered_state_tensor_network, hamiltonian_matrix)]

    parameters_set = extract_parameters_from_tensor_network(state_tensor_network)
    
    for iteration in range(iterations_number):
     
        gradient = vqe_cost_function_gradient_projectq_cheaty(qubits_number, state_tensor_network, hamiltonian_matrix)
        
        parameters_set = parameters_set - gamma*gradient

        update_tensor_network(state_tensor_network, parameters_set)

        daggered_state_tensor_network = daggerize_tensor_network(state_tensor_network)
        
        hamiltonian_expectation_values.append(hamiltonian_matrix_element_part_projectq_cheaty('Re', qubits_number, state_tensor_network, daggered_state_tensor_network, hamiltonian_matrix))

    return hamiltonian_expectation_values


def perform_vqe(qubits_number, hamiltonian_dictionary, state_tensor_network, minimization_method, engine, external_gradient=False, extremum='minimum'):

    if extremum == 'maximum':
        sign = -1
    else:
        sign = 1

    hamiltonian_op = hamiltonian_dictionary_to_operator(qubits_number, hamiltonian_dictionary)

    expectation_values_array = []

    def function(x, qubits_number, state_tensor_network, hamiltonian_dictionary, engine):

        quantum_state = engine.allocate_qureg(qubits_number)

        network_parameters = x

        for block in state_tensor_network:

            parameters_number = len(block['parameters'])
            current_parameters = network_parameters[block['block_number']*parameters_number:(block['block_number']+1)*parameters_number]

            block_unitaries_list = unitary_operator(qubits_number = qubits_number,
                                                    operator_name = block['operator'],
                                                    qubits = block['qubits'],
                                                    parameters = current_parameters,
                                                    daggered = block['daggered'],
                                                    derivatives = block['derivatives'])
            for unitary in block_unitaries_list:
                unitary | quantum_state

        engine.flush()
        expectation_value = engine.backend.get_expectation_value(hamiltonian_op, quantum_state)
        All(Measure) | quantum_state

        expectation_values_array.append(expectation_value)

        return sign*expectation_value

    initial_parameters = []
    for block in state_tensor_network:
        initial_parameters += list(block["parameters"])

    if external_gradient == True:
        def dummy_jac(x, qubits_number, state_tensor_network, hamiltonian_dictionary, engine):
            update_tensor_network(state_tensor_network, x)
            return vqe_cost_function_gradient(qubits_number, state_tensor_network, hamiltonian_dictionary, engine)
        jac = dummy_jac
    else:
        jac = None

    args = (qubits_number, state_tensor_network, hamiltonian_dictionary, engine)

    optimization_result = minimize(function, initial_parameters, args=args, method=minimization_method, jac=jac)

    return optimization_result, expectation_values_array


def perform_vqe_projectq_cheaty(qubits_number, hamiltonian_dictionary, state_tensor_network, minimization_method, engine, external_gradient=False, external_hessian=False, extremum='minimum'):

    if extremum == 'maximum':
        sign = -1
    elif extremum == 'minimum':
        sign = 1

    hamiltonian_matrix = hamiltonian_dictionary_to_matrix(hamiltonian_dictionary)

    expectation_values_array = []

    def function(x, qubits_number, state_tensor_network, hamiltonian_matrix, engine, simulator):

        update_tensor_network(state_tensor_network, x)

        daggered_state_tensor_network = daggerize_tensor_network(state_tensor_network)
        
        expectation_value = hamiltonian_matrix_element_part_projectq_cheaty('Re', qubits_number, state_tensor_network, daggered_state_tensor_network, hamiltonian_matrix, engine, simulator)

        expectation_values_array.append(expectation_value)
        
        return sign*expectation_value

    initial_parameters = []
    for block in state_tensor_network:
        initial_parameters += list(block["parameters"])

    if external_gradient == True:
        def dummy_jac(x, qubits_number, state_tensor_network, hamiltonian_dictionary, engine):
            update_tensor_network(state_tensor_network, x)
            return vqe_cost_function_gradient(qubits_number, state_tensor_network, hamiltonian_dictionary, engine)
        jac = dummy_jac
    else:
        jac = None

    args = (qubits_number, state_tensor_network, hamiltonian_matrix, engine, simulator)

    optimization_result = minimize(function, initial_parameters, args=args, method=minimization_method, jac=jac)

    return optimization_result, expectation_values_array


# NumPy #

def vqe_cost_function_gradient_numpy(qubits_number, state_tensor_network, hamiltonian_dictionary):
    '''
    the same function but for NumPy
    '''
    daggered_state_tensor_network = daggerize_tensor_network(state_tensor_network)

    components = []

    for block in state_tensor_network:

        for parameter_number in range(len(block['parameters'])):

            block['derivatives'].append(parameter_number)

            components.append(hamiltonian_matrix_element_part_numpy('Re', qubits_number, state_tensor_network, daggered_state_tensor_network, hamiltonian_dictionary))

            block['derivatives'].pop(-1)

    return 2*np.array(components)


def vqe_cost_function_hessian_numpy(qubits_number, state_tensor_network, hamiltonian_dictionary):
    '''
    returns the Hessian of the VQE objective function
    '''
    parameters_number_total = tensor_network_parameters_number(state_tensor_network)

    components_1 = np.zeros((parameters_number_total, parameters_number_total))
    components_2 = np.zeros((parameters_number_total, parameters_number_total))

    for l in range(parameters_number_total):
        for k in range(l, parameters_number_total):

            parameters_checked = 0
            for block in state_tensor_network:

                block_parameters_number = len(block['parameters'])

                if l >= parameters_checked and l < parameters_checked+block_parameters_number:
                    block['derivatives'].append(l-parameters_checked)
                    daggered_state_tensor_network = daggerize_tensor_network(state_tensor_network)
                    block['derivatives'].pop(-1)
                if k >= parameters_checked and k < parameters_checked+block_parameters_number:
                    block['derivatives'].append(k-parameters_checked)
                    derivated_block_number_k = block['block_number']

                parameters_checked += block_parameters_number

            component_lk = hamiltonian_matrix_element_part_numpy('Re', qubits_number, state_tensor_network, daggered_state_tensor_network, hamiltonian_dictionary)
            components_1[k][l] = component_lk
            components_1[l][k] = component_lk

            state_tensor_network[derivated_block_number_k]['derivatives'].pop(-1)

    for l in range(parameters_number_total):
        for k in range(l, parameters_number_total):

            daggered_state_tensor_network = daggerize_tensor_network(state_tensor_network)

            parameters_checked = 0
            for block in state_tensor_network:

                block_parameters_number = len(block['parameters'])

                if l >= parameters_checked and l < parameters_checked+block_parameters_number:
                    block['derivatives'].append(l-parameters_checked)
                    derivated_block_number_l = block['block_number']
                if k >= parameters_checked and k < parameters_checked+block_parameters_number:
                    block['derivatives'].append(k-parameters_checked)
                    derivated_block_number_k = block['block_number']

                parameters_checked += block_parameters_number

            component_lk = hamiltonian_matrix_element_part_numpy('Re', qubits_number, state_tensor_network, daggered_state_tensor_network, hamiltonian_dictionary)
            components_2[k][l] = component_lk
            components_2[l][k] = component_lk

            state_tensor_network[derivated_block_number_k]['derivatives'].pop(-1)
            state_tensor_network[derivated_block_number_l]['derivatives'].pop(-1)

    return 2*np.add(np.array(components_1), np.array(components_2))#, components_1, components_2


def perform_vqe_numpy(qubits_number, hamiltonian_dictionary, state_tensor_network, minimization_method, external_gradient=False, external_hessian=False, extremum='minimum', options={}):

    if extremum == 'maximum':
        sign = -1
    else:
        sign = 1

    expectation_values_array = []

    def function(x, qubits_number, state_tensor_network, hamiltonian_dictionary):

        update_tensor_network(state_tensor_network, x)

        daggered_state_tensor_network = daggerize_tensor_network(state_tensor_network)

        expectation_value = hamiltonian_matrix_element_part_numpy('Re', qubits_number, state_tensor_network, daggered_state_tensor_network, hamiltonian_dictionary)

        expectation_values_array.append(expectation_value)

        return sign*expectation_value

    initial_parameters = []
    for block in state_tensor_network:
        initial_parameters += list(block["parameters"])

    if external_gradient == True:
        def dummy_jac(x, qubits_number, state_tensor_network, hamiltonian_dictionary):
            update_tensor_network(state_tensor_network, x)
            return vqe_cost_function_gradient_numpy(qubits_number, state_tensor_network, hamiltonian_dictionary)
        jac = dummy_jac
    else:
        jac = None

    if external_hessian == True:
        def dummy_hess(x, qubits_number, state_tensor_network, hamiltonian_dictionary):
            update_tensor_network(state_tensor_network, x)
            return vqe_cost_function_hessian_numpy(qubits_number, state_tensor_network, hamiltonian_dictionary)
        hess = dummy_hess
    else:
        hess = None

    args = (qubits_number, state_tensor_network, hamiltonian_dictionary)

    optimization_result = minimize(fun=function, x0=initial_parameters, args=args, method=minimization_method, jac=jac, hess=hess, options=options)

    return optimization_result, expectation_values_array


def perform_vqe_gradient_descent_numpy(qubits_number, hamiltonian_dictionary, state_tensor_network, gamma=0.1, iterations_number=20, with_hessian=False):
    '''
    performs dummy gradient descent without termination conditions and with fixed step size \gamma
    '''

    daggered_state_tensor_network = daggerize_tensor_network(state_tensor_network)

    hamiltonian_expectation_values = [hamiltonian_matrix_element_part_numpy('Re', qubits_number, state_tensor_network, daggered_state_tensor_network, hamiltonian_dictionary)]

    parameters_set = extract_parameters_from_tensor_network(state_tensor_network)

    for iteration in range(iterations_number):

        gradient = vqe_cost_function_gradient_numpy(qubits_number, state_tensor_network, hamiltonian_dictionary)

        if with_hessian == True:
            hessian = vqe_cost_function_hessian_numpy(qubits_number, state_tensor_network, hamiltonian_dictionary)
            try:
                hessian_iversed = np.linalg.pinv(hessian + 1e-3*np.eye(len(hessian)))
            except: # if the Hessian is singular
                hessian_iversed = np.linalg.pinv(hessian +  1e-3*np.eye(len(hessian)))
                print('Hessian is singular!')
            parameters_set = parameters_set - np.dot(hessian_iversed, gradient)
        else:
            parameters_set = parameters_set - gamma*gradient

        update_tensor_network(state_tensor_network, parameters_set)

        daggered_state_tensor_network = daggerize_tensor_network(state_tensor_network)

        hamiltonian_expectation_values.append(hamiltonian_matrix_element_part_numpy('Re', qubits_number, state_tensor_network, daggered_state_tensor_network, hamiltonian_dictionary))

    return hamiltonian_expectation_values


def perform_vqe_gradient_descent_adam_numpy(qubits_number, hamiltonian_dictionary, state_tensor_network, iterations_number=20, eta=0.001):
    '''
    performs VQE using the Adam technique with learning rate \eta
    '''
    daggered_state_tensor_network = daggerize_tensor_network(state_tensor_network)

    hamiltonian_expectation_values = [hamiltonian_matrix_element_part_numpy('Re', qubits_number, state_tensor_network, daggered_state_tensor_network, hamiltonian_dictionary)]

    parameters_set = []
    for block in state_tensor_network:
        parameters_set = np.concatenate((parameters_set, block['parameters']))

    m = np.zeros(len(parameters_set))
    v = np.zeros(len(parameters_set))

    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-8

    for iteration in range(iterations_number):

        gradient = vqe_cost_function_gradient_numpy(qubits_number, state_tensor_network, hamiltonian_dictionary)

        t = iteration + 1
        m = beta_1*m + (1 - beta_1)*gradient
        v = beta_2*v + (1 - beta_2)*gradient**2
        m_hat = m/(1 - beta_1**t)
        v_hat = v/(1 - beta_2**t)
        parameters_set = parameters_set - eta*m_hat/(sqrt(v_hat) + epsilon)

        update_tensor_network(state_tensor_network, parameters_set)

        daggered_state_tensor_network = daggerize_tensor_network(state_tensor_network)

        hamiltonian_expectation_values.append(hamiltonian_matrix_element_part_numpy('Re', qubits_number, state_tensor_network, daggered_state_tensor_network, hamiltonian_dictionary))

    return hamiltonian_expectation_values

## VES ##

# ProjectQ #

def eigenvector_cost_function_projectq(qubits_number, state_tensor_network, unitary_tensor_network, engine):
    '''
    returns the value of the cost function of the eigenvector search algorithm
    |<0|U^+ Q U|0>|^2
    '''

    quantum_state = engine.allocate_qureg(qubits_number)

    # prepares the initial state U |00...0>
    for block in state_tensor_network:

        # generate the list of unitaries corresponing to the current block
        block_unitaries_list = unitary_operator(qubits_number = qubits_number,
                                                operator_name = block['operator'],
                                                qubits = block['qubits'],
                                                parameters = block['parameters'],
                                                daggered = block['daggered'],
                                                derivatives = block['derivatives'])

        # apply all the unitaries to the state one by one
        for unitary in block_unitaries_list:
            unitary | quantum_state
    # send all the operations to the backend
    engine.flush()

    # apply the unitary; gives Q U |00...0>
    for block in unitary_tensor_network:

        block_unitaries_list = unitary_operator(qubits_number = qubits_number,
                                                operator_name = block['operator'],
                                                qubits = block['qubits'],
                                                parameters = block['parameters'],
                                                daggered = block['daggered'],
                                                derivatives = block['derivatives'])

        for unitary in block_unitaries_list:
            unitary | quantum_state
    engine.flush()

    # apply the daggered state tensor network; gives U^+ Q U |00...0>
    daggered_state_tensor_network = daggerize_tensor_network(state_tensor_network)
    for block in daggered_state_tensor_network:

        block_unitaries_list = unitary_operator(qubits_number = qubits_number,
                                                operator_name = block['operator'],
                                                qubits = block['qubits'],
                                                parameters = block['parameters'],
                                                daggered = block['daggered'],
                                                derivatives = block['derivatives'])

        for unitary in block_unitaries_list:
            unitary | quantum_state
    engine.flush()

    wanted_outcome = ['0' * qubits_number]
    probability_0 = engine.backend.get_probability(wanted_outcome, quantum_state)

    # deallocate the quantum register
    All(Measure) | quantum_state
    engine.flush()

    return probability_0

def eigenvector_cost_function_gradient_projectq(qubits_number, state_tensor_network, unitary_tensor_network, engine):
    '''
    returns the gradient of the cost function of eigenvector search algorithm as a NumPy vector
    '''

    daggered_state_tensor_network = daggerize_tensor_network(state_tensor_network)
    daggered_unitary_tensor_network = daggerize_tensor_network(unitary_tensor_network)

    concatenated_tensor_network = sum_tensor_networks(state_tensor_network, unitary_tensor_network, daggered_state_tensor_network)
    term_1_1 = unitary_expectation_value_projectq(qubits_number, concatenated_tensor_network, engine)
    term_2_1 = term_1_1.conjugate()

    components = []

    for block in state_tensor_network:

        for parameter_number in range(len(block['parameters'])):

            block['derivatives'].append(parameter_number)

            concatenated_tensor_network = sum_tensor_networks(state_tensor_network, daggered_unitary_tensor_network, daggered_state_tensor_network)
            term_1_2 = unitary_expectation_value_projectq(qubits_number, concatenated_tensor_network, engine)

            concatenated_tensor_network = sum_tensor_networks(state_tensor_network, unitary_tensor_network, daggered_state_tensor_network)
            term_2_2 = unitary_expectation_value_projectq(qubits_number, concatenated_tensor_network, engine)

            block['derivatives'].pop(-1)

            components.append(2*np.real(term_1_1*term_1_2) + 2*np.real(term_2_1*term_2_2))

    return np.array(components)


def perform_eigenvector_gradient_descent_projectq(qubits_number, state_tensor_network, unitary_tensor_network, engine, gamma=0.1, iterations_number=20):
    '''
    performs dummy gradient descent without termination conditions and with fixed step size \gamma
    '''

    # the first step
    cost_function_values = [eigenvector_cost_function_projectq(qubits_number, state_tensor_network, unitary_tensor_network, engine)]

    parameters_number_total = 0
    for block in state_tensor_network:
        parameters_number_total += len(block['parameters'])

    gradient_previuos = np.array([0 for i in range(parameters_number_total)])
    parameters_set_previous = np.array([0 for i in range(parameters_number_total)])

    for iteration in range(iterations_number):

        print('Iteration', iteration)

        gradient_current = eigenvector_cost_function_gradient_projectq(qubits_number, state_tensor_network, unitary_tensor_network, engine)
        gradient_difference = gradient_current - gradient_previuos

        parameters_set_current = []
        for block in state_tensor_network:
            parameters_set_current = np.concatenate((parameters_set_current, block['parameters']))
        parameters_set_difference = parameters_set_current - parameters_set_previous

        parameters_set_next = parameters_set_current + gamma*gradient_current

        parameters_updated = 0
        for block in state_tensor_network:
            parameters_number = len(block['parameters'])
            block['parameters'] = parameters_set_next[parameters_updated : parameters_updated+parameters_number]
            parameters_updated += parameters_number

        cost_function_values.append(eigenvector_cost_function_projectq(qubits_number, state_tensor_network, unitary_tensor_network, engine))

        gradient_previous = gradient_current
        parameters_set_previous = parameters_set_current

    print("Completed.")

    return cost_function_values

# NumPy #

def eigenvector_cost_function_numpy(qubits_number, state_tensor_network, unitary_tensor_network, engine):
    '''
    the same function but for NumPy
    '''

    daggered_state_tensor_network = daggerize_tensor_network(state_tensor_network)
    concatenated_tensor_network = sum_tensor_networks(state_tensor_network, unitary_tensor_network, daggered_state_tensor_network)

    cost_function_value = unitary_expectation_value_numpy(qubits_number, concatenated_tensor_network, engine)

    return abs(cost_function_value)**2


def eigenvector_cost_function_gradient_numpy(qubits_number, state_tensor_network, unitary_tensor_network, engine):

    daggered_state_tensor_network = daggerize_tensor_network(state_tensor_network)
    daggered_unitary_tensor_network = daggerize_tensor_network(unitary_tensor_network)

    concatenated_tensor_network = sum_tensor_networks(state_tensor_network, unitary_tensor_network, daggered_state_tensor_network)
    term_1_1 = unitary_expectation_value_numpy(qubits_number, concatenated_tensor_network, engine)
    term_2_1 = term_1_1.conjugate()

    components = []

    for block in state_tensor_network:

        for parameter_number in range(len(block['parameters'])):

            block['derivatives'].append(parameter_number)

            concatenated_tensor_network = sum_tensor_networks(state_tensor_network, daggered_unitary_tensor_network, daggered_state_tensor_network)
            term_1_2 = unitary_expectation_value_numpy(qubits_number, concatenated_tensor_network, engine)

            concatenated_tensor_network = sum_tensor_networks(state_tensor_network, unitary_tensor_network, daggered_state_tensor_network)
            term_2_2 = unitary_expectation_value_numpy(qubits_number, concatenated_tensor_network, engine)

            block['derivatives'].pop(-1)

            components.append(2*np.real(term_1_1*term_1_2) + 2*np.real(term_2_1*term_2_2))

    return np.array(components)


def perform_eigenvector_gradient_descent_numpy(qubits_number, state_tensor_network, unitary_tensor_network, engine, gamma=0.1, iterations_number=20):

    # the first step
    cost_function_values = [eigenvector_cost_function_numpy(qubits_number, state_tensor_network, unitary_tensor_network, engine)]

    parameters_number_total = 0
    for block in state_tensor_network:
        parameters_number_total += len(block['parameters'])

    gradient_previuos = np.array([0 for i in range(parameters_number_total)])
    parameters_set_previous = np.array([0 for i in range(parameters_number_total)])

    for iteration in range(iterations_number):

        print('Iteration', iteration)

        gradient_current = eigenvector_cost_function_gradient_numpy(qubits_number, state_tensor_network, unitary_tensor_network, engine)
        gradient_difference = gradient_current - gradient_previuos

        parameters_set_current = []
        for block in state_tensor_network:
            parameters_set_current = np.concatenate((parameters_set_current, block['parameters']))
        parameters_set_difference = parameters_set_current - parameters_set_previous

        parameters_set_next = parameters_set_current + gamma*gradient_current

        parameters_updated = 0
        for block in state_tensor_network:
            parameters_number = len(block['parameters'])
            block['parameters'] = parameters_set_next[parameters_updated : parameters_updated+parameters_number]
            parameters_updated += parameters_number

        cost_function_values.append(eigenvector_cost_function_numpy(qubits_number, state_tensor_network, unitary_tensor_network, engine))

        gradient_previous = gradient_current
        parameters_set_previous = parameters_set_current

    print("Completed.")

    return cost_function_values
