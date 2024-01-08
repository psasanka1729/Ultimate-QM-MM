#!/usr/bin/env python
# coding: utf-8

# ## This program solves the single qubit Lindblad problem

# In[1]:


#import os
import sys
import numpy as np
#import matplotlib.pyplot as plt
import qiskit
from qiskit import*


# In[2]:


""" Matrices required for calculations.   """
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

def commutator(matrix_a,matrix_b):
    return matrix_a@matrix_b - matrix_b@matrix_a
def anti_commutator(matrix_a,matrix_b):
    return matrix_a@matrix_b + matrix_b@matrix_a

# partial trace of a 4x4 matrix.
def partial_trace(mat):
    return np.matrix([[mat[0,0]+mat[1,1],mat[0,2]+mat[1,3]],
                      [mat[2,0]+mat[3,1],mat[2,2]+mat[3,3]]])

# analytical solution from literature.
def rho_SS_analytical(yss_p,zss_p):
    return (1/2)*(I2+yss_p*sigma_y+zss_p*sigma_z)

# density matrix in terms of angles
def rho_SS_from_angles(theta_x,theta_y,theta_z,theta_ry):
    rho_11 = (1/2)*(1+np.cos(theta_x)*np.cos(theta_y))
    rho_12 = ((1/2)*np.exp(-1j*theta_z)*np.cos(theta_ry/2)*(np.sin(theta_y)*np.cos(theta_x)
                                                           +1j*np.sin(theta_x)))
    return np.matrix([[rho_11,rho_12],[rho_12.conj(),1-rho_11]])


# In[3]:


# parameters of the Hamiltonian and the Lindbladians.
b = 0.9
gamma_1 = 0.3
gamma_2 = 0.4

def lindblad_equation(density_mat,lindbladian_lst,hamiltonian = -(b/2)*sigma_x,):
    s = -1j*commutator(hamiltonian,density_mat)
    for lindbladian in lindbladian_lst:
        s += (lindbladian@density_mat@(lindbladian.conj().T)
                -1/2*anti_commutator((lindbladian.conj().T)@lindbladian,density_mat))
    return s

# known analytical solution
Delta = 2*b**2+gamma_1**2+4*gamma_1*gamma_2
yss= (2*b*gamma_1)/Delta
zss = (gamma_1*(gamma_1+4*gamma_2))/Delta

theta_y_exact = 0.0#np.pi/6 # np.pi/8
theta_x_exact = np.arccos(zss/np.cos(theta_y_exact))
theta_z_exact = np.arctan(-np.sin(theta_y_exact)/np.tan(theta_x_exact))
theta_ry_exact = 2*np.arccos(-yss*np.cos(theta_z_exact)/np.sin(theta_x_exact))


# In[4]:


# quantum circuit to perform shadow tomography
def test_quantum_circuit(theta_x,theta_y,theta_z,theta_ry):
    qc_test = QuantumCircuit(2)
    qc_test.rx(theta_x,1)
    qc_test.ry(theta_y,1) 
    qc_test.rz(theta_z,1)    
    qc_test.cry(theta_ry,1,0)
    return qc_test


# In[5]:


""" Inverse shadow channel. """
def Minv(N_qubits,X):
    return ((2**N_qubits+1.))*X - np.eye(2**N_qubits)


def shadow_tomography_clifford(N,K,N_qubits,quantum_circuit):

    n_Shadows = N*K # number of shadows
    
    # generating a set of Clifford gates
    cliffords = [qiskit.quantum_info.random_clifford(N_qubits-1,) for _ in range(n_Shadows)]
    
    # exact density matrix from the quantum circuit
    rho_actual = qiskit.quantum_info.DensityMatrix(quantum_circuit).data
    
    # list to store the results from the classical measurements
    results = []
    
    for cliff in cliffords:
        # appends the Clifford circuit to the original circuit
        qc_c  = quantum_circuit.compose(cliff.to_circuit(),qubits=[1])
        # measures the qubits
        c = ClassicalRegister(1)
        qc_c.add_register(c)
        qc_c.measure(1, c[0])
        simulator = Aer.get_backend("qasm_simulator")
        job = execute(qc_c, simulator, shots=1)
        result = job.result()
        counts = result.get_counts(qc_c)
        #counts = qiskit.quantum_info.Statevector(qc_c).sample_counts(reps)
        results.append(counts)
        
    # list to store the shadow density matrix    
    shadows = []
    for cliff, res in zip(cliffords, results):
        # Clifford gate is converted into a matrix.
        mat    = cliff.adjoint().to_matrix()
        for bit,count in res.items():
            Ub = mat[:,int(bit,2)] # this is Udag|b>           
            shadows.append(Minv(N_qubits-1,np.outer(Ub,Ub.conj()))*count)

    return partial_trace(rho_actual),shadows,qc_c

# In[6]:


def linear_function_prediction(N,K,operator_linear,list_of_shadows):
    list_of_means = []
    
    # calculating K means
    for k in range(1,K+1):
        shadows_mean = 0.0
        for j in range(N*(k-1)+1,N*k+1):
            rho_j = list_of_shadows[j-1]
            shadows_mean += np.trace(operator_linear@rho_j )
            
        list_of_means.append(shadows_mean/N)
        
    # calculating the median of K means
    return np.median(list_of_means)

def quadratic_function_prediction(N,K,operator_m,operator_n,list_of_shadows):
    
    list_of_means = []
    
    SWAP = np.matrix([[1,0,0,0],
                      [0,0,1,0],
                      [0,1,0,0],
                      [0,0,0,1]])
    r"""
    
    calculating the operator O. Note that the order of operator_m and operator_n is irrelevant.
    
    """
    
    O_quadratic = SWAP@np.kron(operator_m,operator_n)
    
    # calculating K means
    for k in range(1,K+1):
        shadows_mean = 0.0        
        for j in range(N*(k-1)+1,N*k+1):
            for l in range(N*(k-1)+1,N*k+1):
                if j != l:
                    rho_j = list_of_shadows[j-1]
                    rho_l = list_of_shadows[l-1]
                    shadows_mean += np.trace(O_quadratic@np.kron(rho_j,rho_l))
                    
        list_of_means.append(shadows_mean/(N*(N-1)))
        
    # calculating their median
    return np.median(list_of_means)


# In[ ]:


N = 80
K = 10
N_qubits = 2
n_Shadows = N*K
number_of_angle_divisons = 100
angles_lst = np.linspace(-np.pi,np.pi,number_of_angle_divisons)
repetitions = int(sys.argv[1])

cost_function_one_rep = []

for angle_index in range(len(angles_lst)):

    quadratic_function_i_i_lst = []
    quadratic_function_y_i_lst = [] 
    quadratic_function_z_i_lst = []
    quadratic_function_x_x_lst = []
    quadratic_function_z_z_lst = [] 
    quadratic_function_y_z_lst = []

    st_instance = shadow_tomography_clifford(N,K,N_qubits,test_quantum_circuit(angles_lst[angle_index],
                                                        theta_y_exact,theta_z_exact,theta_ry_exact))
    #np.save("exact_density_matrix_"+str(N*K)+"_"+str(repetitions)+"_"+str(angle_index)+".npy",st_instance[0])    
    #np.save("shadow_density_matrix_"+str(N*K)+"_"+str(repetitions)+"_"+str(angle_index)+".npy",st_instance[1])

    shadows_lst = st_instance[1]

    quadratic_function_i_i_lst.append(quadratic_function_prediction(N,K,I2,I2,shadows_lst))
    quadratic_function_y_i_lst.append(quadratic_function_prediction(N,K,sigma_y,I2,shadows_lst))
    quadratic_function_z_i_lst.append(quadratic_function_prediction(N,K,sigma_z,I2,shadows_lst))
    quadratic_function_x_x_lst.append(quadratic_function_prediction(N,K,sigma_x,sigma_x,shadows_lst))
    quadratic_function_z_z_lst.append(quadratic_function_prediction(N,K,sigma_z,sigma_z,shadows_lst))
    quadratic_function_y_z_lst.append(quadratic_function_prediction(N,K,sigma_y,sigma_z,shadows_lst))

    quadratic_function_i_i_lst = np.array(quadratic_function_i_i_lst)
    quadratic_function_y_i_lst = np.array(quadratic_function_y_i_lst) 
    quadratic_function_z_i_lst = np.array(quadratic_function_z_i_lst)
    quadratic_function_x_x_lst = np.array(quadratic_function_x_x_lst)
    quadratic_function_z_z_lst = np.array(quadratic_function_z_z_lst)
    quadratic_function_y_z_lst = np.array(quadratic_function_y_z_lst) 

    cost_function_one_rep.append(((b**2/2)*(-quadratic_function_x_x_lst+quadratic_function_i_i_lst)
	    + (1j*b*gamma_1/4)*(4*1j*quadratic_function_y_i_lst-2*1j*quadratic_function_y_z_lst)
	    -2*gamma_2*b*(quadratic_function_y_z_lst)
	    +(gamma_1**2/16)*(10*quadratic_function_i_i_lst+6*quadratic_function_z_z_lst-16*quadratic_function_z_i_lst)
	    +(gamma_1*gamma_2/2)*(-2*quadratic_function_z_z_lst+2*quadratic_function_i_i_lst)
	    +(2*gamma_2**2)*(quadratic_function_i_i_lst-quadratic_function_z_z_lst))[0])

np.save("cost_function_matrix.npy",cost_function_one_rep)

