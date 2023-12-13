#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/psasanka1729/Ultimate-QM-MM/blob/main/shadow_tomography_qiskit_code.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[2]:


import numpy as np
#import matplotlib.pyplot as plt
import qiskit
from qiskit.quantum_info import Statevector
from qiskit import*


# ## Pauli matrices

# In[3]:


I2 = np.matrix([[1,0],
                [0,1]])
hadamard_gate  = np.matrix([[1,1],
                [1,-1]])* (1/np.sqrt(2))
phase_gate  = np.matrix([[1,0],
                [0,1j]])
sigma_x  = np.matrix([[0,1],
                [1,0]])
sigma_y  = np.matrix([[0,-1j],
                [1j,0]])
sigma_z  = np.matrix([[1,0],
                [0,-1]])


# In[4]:


def commutator(matrix_a,matrix_b):
    return matrix_a@matrix_b - matrix_b@matrix_a
def anti_commutator(matrix_a,matrix_b):
    return matrix_a@matrix_b + matrix_b@matrix_a
# partial trace of a 4x4 matrix.
def partial_trace(mat):
    return np.matrix([[mat[0,0]+mat[1,1],mat[0,2]+mat[1,3]],
                      [mat[2,0]+mat[3,1],mat[2,2]+mat[3,3]]])

"""

"""
b = 0.1
def lindblad_equation(density_mat,lindbladian_lst,hamiltonian = -(b/2)*sigma_x,):
    s = -1j*commutator(hamiltonian,density_mat)
    for lindbladian in lindbladian_lst:
        s += (lindbladian@density_mat@(lindbladian.conj().T)
                -1/2*anti_commutator((lindbladian.conj().T)@lindbladian,density_mat))
    return s


# ## Shadow tomography

# In[5]:


# returns N*K classical shadows.
def classical_shadow_using_ST(N,K,quantum_circuit):

    # exact density matrix from the quantum circuit.
    sv = np.matrix(Statevector.from_instruction(quantum_circuit)).T
    rho_exact = sv@(sv.conj().T)    
    
    # number of shadows
    nShadows = N*K
    
    number_of_shots = 1
    #rng = np.random.default_rng(1000)

    # number of qubits the Clifford gates will be applied to.
    N_clifford_qubit = 1
    cliffords = [qiskit.quantum_info.random_clifford(N_clifford_qubit) for _ in range(nShadows)]
    results = []

    r"""
     creating a empty circuit to compose to the Clifford circuit
     so that it becomes the same dimension as the original circuit."""
    for cliff in cliffords:
        # converting the clifford gates to circuit and appending it to the original circuit.
        qc_c  = cliff.to_circuit()
        quantum_circuit  = quantum_circuit.compose(qc_c,qubits=[1])
        quantum_circuit.measure(1,0)
        # execute the quantum circuit
        backend = BasicAer.get_backend('qasm_simulator') # the device to run on
        result = backend.run(transpile(quantum_circuit, backend), shots=number_of_shots).result()
        counts  = result.get_counts(quantum_circuit)

        results.append(counts)

    # performing rho_hat = (2**N+1)*(U^{\dagger}|b><b|U-I).
    shadow_lst = []
    # loops over the clifford gates and their corresponding measurement results.
    for cliff, res in zip(cliffords, results):
        # U^{\dagger} matrix.
        mat    = cliff.adjoint().to_matrix()
        for bit,count in res.items():
            try:
                bit = counts["0"]
            except KeyError:
                bit = counts["1"]

            if bit == "0":
                # |0>
                ket_b = np.matrix([[1],[0]])
            else:
                # |1>
                ket_b = np.matrix([[0],[1]])

            Ub = mat@ket_b

            shadow_lst.append(((2**N_clifford_qubit+1.))*np.outer(Ub,(Ub.conj().T))*count - np.eye(2**N_clifford_qubit))
            
    return partial_trace(rho_exact), shadow_lst


# ## Variation of parameters for single qubit

# In[6]:


b = 0.2
gamma_1 = 0.1
gamma_2 = 0.9

# known analytical solution
Delta = 2*b**2+gamma_1**2+4*gamma_1*gamma_2
yss= (2*b*gamma_1)/Delta
zss = (gamma_1*(gamma_1+4*gamma_2))/Delta

theta_y_exact = np.pi/8
theta_x_exact = np.arccos(zss/np.cos(theta_y_exact))
theta_z_exact = np.arctan(-np.sin(theta_y_exact)/np.tan(theta_x_exact))
theta_ry_exact = 2*np.arccos(-yss*np.cos(theta_z_exact)/np.sin(theta_x_exact))

# ## Median of means estimator for linear and non-linear function

# In[7]:


# linear function
def linear_o_k(O_i,k,shadow_lst):
    s = 0.0
    for j in range(N*(k-1)+1,N*k+1):
        rho_j = shadow_lst[j-1]
        s += np.trace(O_i@rho_j)
    return s/N

def linear_o(O_i,shadow_lst):
    linear_o_k_lst = []
    for kk in range(1,K+1):
        linear_o_k_lst.append(linear_o_k(O_i,kk,shadow_lst))
    return np.median(linear_o_k_lst)

# quadratic function
SWAP_matrix = np.matrix([[1,0,0,0],
                  [0,0,1,0],
                  [0,1,0,0],
                  [0,0,0,1]])

# median of means method to estimate the cost function
def quadratic_o_k(k,X_i,X_j,shadow_lst):
    # f(\rho) = \tr(X_i rho X_j \rho)
    O_i = SWAP_matrix@np.kron(X_i,X_j)    
    s = 0.0
    for j in range(N*(k-1)+1,N*k+1):
        for l in range(N*(k-1)+1,N*k+1):
            if j != l:
                rho_j = shadow_lst[j-1]
                rho_l = shadow_lst[l-1]
                # cost function
                s += np.trace(O_i@(np.kron(rho_j,rho_l)))
    return s/(N*(N-1))

# following function returns the estimated cost function
def quadratic_o(X_i,X_j,shadow_lst):
    o_k_lst = []
    for kk in range(1,K+1):
        o_k_lst.append(quadratic_o_k(kk,X_i,X_j,shadow_lst))
    return np.median(o_k_lst)


# ## Shadow tomography

# In[8]:


# quantum circuit to perform shadow tomography
def test_quantum_circuit(theta_x,theta_y,theta_z,theta_ry):
    qc_test = QuantumCircuit(2,1)
    qc_test.rx(theta_x,0)
    qc_test.ry(theta_y,0) 
    qc_test.rz(theta_z,0)    
    qc_test.cry(theta_ry,0,1)
    return qc_test


# In[10]:



# In[ ]:


divisons = 10
angles_lst = np.linspace(-np.pi,np.pi,divisons)

def cost_function(rho):
    return np.trace(rho@rho)

exact_density_matrix_lst  = []
shadow_density_matrix_lst = []
N = 300
K = 10
for jj in range(divisons):
    print(jj)
    exact_and_shadows_lst = classical_shadow_using_ST(N,K,test_quantum_circuit(angles_lst[jj],
                                               theta_y_exact,
                                               theta_z_exact,
                                               theta_ry_exact))
  
    exact_density_matrix_lst.append(exact_and_shadows_lst[0])
    shadow_density_matrix_lst.append(exact_and_shadows_lst[1])


# In[ ]:


np.save("exact_density_matrix_lst_"+str(N*K)+".npy",exact_density_matrix_lst)    
np.save("shadows_lst_"+str(N*K)+".npy",shadow_density_matrix_lst)


# ## Plot

# In[ ]:


rho_exact_file = np.load("exact_density_matrix_lst_"+str(N*K)+".npy")
rho_shadow_file = np.load("shadows_lst_"+str(N*K)+".npy")