# %%
import os
import re
import itertools
import numpy as np
from qiskit import*
# Import from Qiskit Aer noise module
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (NoiseModel, QuantumError, ReadoutError,
    pauli_error, depolarizing_error, thermal_relaxation_error)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.sparse import csr_matrix, kron

large = 40; med = 30; small = 20
params = {'axes.titlesize': med,
          'axes.titlepad' : med,
          'legend.fontsize': med,
          'axes.labelsize': med ,
          'axes.titlesize': med ,
          'xtick.labelsize': med ,
          'ytick.labelsize': med ,
          'figure.titlesize': med}
plt.rcParams['text.usetex'] = True
plt.rcParams.update(params)

# %%
# working directory
path =("/Users/sasankadowarah/Ultimate_QM_MM/")

# parameters for Lindbladians
GAMMA_IN  = 1.6
GAMMA_OUT = 1.6

# number of shots to be used in simulation
NUMBER_OF_SHOTS = 8192

# Identity and Pauli matrices
I2      = np.array([[1, 0], [0, 1]])
SIGMA_X = np.array([[0, 1], [1, 0]])
SIGMA_Y = np.array([[0, -1j], [1j, 0]])
SIGMA_Z = np.array([[1, 0], [0, -1]])

# Dictionary of Pauli matrices
PAULI_MATRICES_DICT = {"I": I2, "X": SIGMA_X, "Y": SIGMA_Y, "Z": SIGMA_Z}

# Matrix for projection to |0> and |1> states
PI_0 = np.matrix([[1,0],[0,0]]) # |0><0|
PI_1 = np.matrix([[0,0],[0,1]]) # |1><1|                         

def controlled_ry_gate_matrix(angle_theta):
        def ry_matrix(angle):
                return np.matrix([[np.cos(angle/2),-np.sin(angle/2)],
                                  [np.sin(angle/2),np.cos(angle/2)]])
        return kron(PI_0,I2) + kron(PI_1,ry_matrix(angle_theta))

def controlled_not_gate_matrix():
        return kron(I2,PI_0) + kron(SIGMA_X,PI_1)

# %% [markdown]
# #### Acetylene molecule Hamiltonian

# %%
# Define a function to convert the second column to complex numbers
def complex_converter(s):
    return complex(s.decode('utf-8'))

# loads the Jordan Wigner transformed Hamiltonian
os.chdir(path)
data = np.loadtxt("acetylene_reduced_bk_hamiltonian.txt", delimiter='\t', converters={1: complex_converter}, dtype=object)

# Separate the data into two arrays
hamiltonian_pauli_lst, hamiltonian_pauli_coeff_lst = data.T

NUMBER_OF_QUBITS = len(str(hamiltonian_pauli_lst[0]))
#NUMBER_OF_QUBITS = 8

# %% [markdown]
# #### Pauli string to nearest neighbor

# %%
r"""
The following function takes a string of Pauli matrices and rearranges it such that
all the "I" matrices are at the both ends of the string. This results in all the other
Pauli matrices "X", "Y" and "Z" to be grouped together which can be implemented using
the properties HZH = X and S^{\dagger}HZHS = Y and a network of CNOT and Rz gate.

Input : "YIXIIZIZIX"
Output: "IIIYXZZXII"

It will be perfomed in the following steps:

        Step 1: Remove all the "I" element from the input Pauli string.
        Step 2: Record the index of the "I" matrices in the input string in step 1.
        Step 3: The "I" matrices are put at the left and the right end of
                the string without the "I" string from step 1. To minimize the
                number of SWAP gates used in the circuit, if the original index
                of the "I" is less than L/2 (L = length of the input Pauli string)
                then the "I" is appended from left and otherwise from the right.
"""

def rearrange_pauli_string_using_nearest_neighbor_swap(initial_pauli_string):
        
        # reversing the Pauli string because Qiskit labels the qubits from right to left
        initial_pauli_string = initial_pauli_string[::-1]

        # Step 1 : removes the I from the Pauli string
        def remove_I(s):
                return s.replace("I", "")
        pauli_string_without_I = remove_I(initial_pauli_string)


        # Step 2
        r""" 
        The following clusters the strings with "I"s. For example, for the string
        "IYIXZXIIZI" it will result [("I",0), ("I",2),("II",6),("I",9)].
        """
        I_string_matches = re.finditer("I+", initial_pauli_string)
        I_string_groups = []
        for match in I_string_matches:
                I_string_groups.append((match.group(), match.start()))      

        # Step 3
        # start with the Pauli string without I        
        rearranged_pauli_string = pauli_string_without_I[::-1] 

        r"""
        The following moves the I's to the left and right end of the string.
        """
        for grouped_i_string in I_string_groups:
                I_string, I_string_index = grouped_i_string
                if I_string_index <= len(initial_pauli_string)/2:
                        rearranged_pauli_string = I_string + rearranged_pauli_string
                else:
                        rearranged_pauli_string = rearranged_pauli_string + I_string  

        return rearranged_pauli_string                       

# %% [markdown]
# #### Pauli string to quantum circuit

# %%
r"""
The following function returns the quantum circuit implementing a given string with
Pauli matrices. It is done in the following steps:

        Step 1: Rearrange the input string such that the "I" matrices are at the
                end of the string. This is done using the function defined above.
        Step 2: Record the indices of the "I" strings in the input and the rearranged
                string.
        Step 3: Initialize the quantum circuit. Put SWAP gates between the old and
                the new indices generated in the previous step.
        Step 4: All Pauli gates other than I should be grouped together now. 
                Implement this group using the properties HZH = X and
                SHZHS^{\dagger} = Y and a network of CNOT and Rz gate.
                Ref: Simulation of electronic structure Hamiltonians using quantum computers,
                James D. Whitfield, Jacob Biamonte, Alán Aspuru-Guzik,
                https://arxiv.org/abs/1001.3855,
                https://www.tandfonline.com/doi/abs/10.1080/00268976.2011.552441
"""

def pauli_string_to_quantum_circuit(initial_pauli_string, initial_pauli_string_coefficient, time_step):

        # Step 1
        # reversing the Pauli string because Qiskit labels the qubits from right to left

        initial_pauli_string = initial_pauli_string[::-1]
        rearranged_pauli_string = rearrange_pauli_string_using_nearest_neighbor_swap(initial_pauli_string)[::-1]

        # Step 2
        def generate_indices(s, chars):
                return {char: [i for i, c in enumerate(s) if c == char] for char in chars}
        # indices of "I" matrices in the input string
        old_indices_I = generate_indices(initial_pauli_string,["I"])["I"]
        # indices of "I" matrices in the rearranged string
        new_indices_I = generate_indices(rearranged_pauli_string,["I"])["I"]

        # Step 3
        pauli_qc = QuantumCircuit(NUMBER_OF_QUBITS)

        swap_qubits_circuit = QuantumCircuit(NUMBER_OF_QUBITS)
        if old_indices_I != new_indices_I: # SWAP is needed if indices are changed

                for k in range(len(new_indices_I)):

                        ni = new_indices_I[k]
                        oi = old_indices_I[k]

                        if ni != oi:

                                r"""
                                The order of the SWAP gates are as follows:
                                for example, if qubit 4 and qubit 0 needs 
                                to be swapped, then the order of the swap is
                                4 <-> 3 <-> 2 <-> 1 <-> 0.
                                The following loop performs this action. 
                                """

                                for i in range(max(ni, oi)-1, min(ni, oi)-1, -1):
                                        swap_qubits_circuit.swap(i+1,i)

        # qubits are swapped before implementing the Pauli gates
        pauli_qc = pauli_qc.compose(swap_qubits_circuit)

        # Step 4
        r"""
        All Pauli gates other than I should be now grouped together. This group can
        be implemented using a network of CNOT and rz gates and by using the basis
        transformation HZH = X and SHZHS^{\dagger} = Y wherever necessary. The following
        records the index of the start and end qubit of this non-identity Pauli group
        and the network is implemented between these two qubits.
        """
        # starting qubit of the non-identity Pauli gates group
        n_zz_gate_start_qubit = next((i for i, char in enumerate(rearranged_pauli_string) if char != "I"), None)      
        # end qubit of the non-identity Pauli gates group
        n_zz_gate_end_qubit = len(rearranged_pauli_string) - next((i for i, char in enumerate(reversed(rearranged_pauli_string)) if char != "I"), None) - 1                 

        r"""
        The following loop performs the necessary basis change to implement X and Y gates.
        """
        for i in range(NUMBER_OF_QUBITS):
                pauli_gate = rearranged_pauli_string[i]
                if   pauli_gate == "X":
                        pauli_qc.h(i)
                elif pauli_gate == "Y":
                        pauli_qc.s(i)
                        pauli_qc.h(i)

        r"""
        The following implements the quantum circuit to perform

                U = exp(-1j*\alpha*dt*Z \otimes Z \otimes Z \otimes ... \otimes Z)

        using CNOT and Rz gates.                
        """

        # cx gates on the left side of the rz gate
        for q in range(n_zz_gate_start_qubit, n_zz_gate_end_qubit):
                pauli_qc.cx(q,q+1)
        # rz gate in the middle of the CNOT gates
        pauli_qc.rz(2*time_step*initial_pauli_string_coefficient, n_zz_gate_end_qubit)
        # cx gates on the right side of the rz gate
        for q in range(n_zz_gate_end_qubit,n_zz_gate_start_qubit,-1):
                pauli_qc.cx(q-1,q)

        r"""
        The following loop performs the necessary basis change to implement X and Y gates.
        """
        for i in range(NUMBER_OF_QUBITS):
                pauli_gate = rearranged_pauli_string[i]
                if   pauli_gate == "X":
                        pauli_qc.h(i)
                elif pauli_gate == "Y":
                        pauli_qc.h(i)
                        pauli_qc.sdg(i)

        # qubits are swapped back to put them in their original order
        pauli_qc = pauli_qc.compose(swap_qubits_circuit)                                                                                              
        return pauli_qc

# %% [markdown]
# #### Quantum circuit for a single Pauli string

# %%
# initial_pauli_string = hamiltonian_pauli_lst[5]
# print(initial_pauli_string)
# print(rearrange_pauli_string_using_nearest_neighbor_swap(initial_pauli_string))
# pauli_string_to_quantum_circuit(initial_pauli_string, 0.1, 0.1).draw("mpl", style = "iqp", scale = 1)

# %% [markdown]
# #### One time step circuit for the molecule

# %%
def one_time_step_circuit(time_step):

     qr = QuantumRegister(NUMBER_OF_QUBITS,"q")
     anc = QuantumRegister(1,r"\rm{ancilla}")

     qc_one_time_step = QuantumCircuit(anc,qr)
     for i in range(1,len(hamiltonian_pauli_lst)):
          qc_one_time_step = qc_one_time_step.compose(pauli_string_to_quantum_circuit(hamiltonian_pauli_lst[i], hamiltonian_pauli_coeff_lst[i].real, 1),qubits = [q for q in range(1,NUMBER_OF_QUBITS+1)])

     # L_out
     theta_out = 2*np.arcsin(np.sqrt(time_step*GAMMA_OUT))
     qc_one_time_step.initialize([1,0],anc[0])
     qc_one_time_step.cry(theta_out,qr[0],anc)
     qc_one_time_step.cx(anc,qr[0])
     qc_one_time_step.initialize([1,0],anc)        

     # L_in
     qc_one_time_step.x(qr[0])
     theta_in = 2*np.arcsin(np.sqrt(time_step*GAMMA_IN))
     qc_one_time_step.cry(theta_in,qr[0],anc)
     qc_one_time_step.cx(anc,qr[0])
     qc_one_time_step.x(qr[0])
     qc_one_time_step.initialize([1,0],anc)     

     return qc_one_time_step
#one_time_step_circuit(0.1).draw("mpl", style = "iqp", scale = 1)

# %% [markdown]
# #### Trotterized circuit for molecule

# %%
def time_evolved_density_matrix(time_step,final_time,initial_state):

     number_of_iterations = int(final_time/time_step)


     transpiled_one_step_circuit = transpile(one_time_step_circuit(time_step), basis_gates = ["rz","cx","h",], optimization_level=2)

     # existing quantum registers
     qr = QuantumRegister(NUMBER_OF_QUBITS,"q")
     anc = QuantumRegister(1,"ancilla")

     # add a classical register
     cr = ClassicalRegister(L,"c")
     # create a new quantum circuit with the classical register
     qc = QuantumCircuit(anc,qr, cr)

     # system is initialized at the user defined state
     qubit_dict = {"1":[0,1],"0":[1,0]}

     r"""
          The initial state is assigned in reverse order because of qiskit's convention
          of counting from right to left. So the first value of the initial state string
          is assigned to the bottom qubit and the last value of the initial state string
          is assigned to the top qubit!
     """
     for s in range(NUMBER_OF_QUBITS):
          qc.initialize(qubit_dict[initial_state[NUMBER_OF_QUBITS-s-1]],qr[s])

     for _ in range(number_of_iterations):
          qc = qc.compose(transpiled_one_step_circuit)

     return qc     

# %% [markdown]
# #### Current operators simplication

# %%
r"""
This function returns the product of any two given Pauli matrices.
Input: pauli_matrix_1;pauli_matrix_2
Output: coefficent in front, product of Pauli matrix 1 and 2
for example for X and Y it will return the pair 1j,Z.
"""

def pauli_product(pauli_i, pauli_j):
    
        # multiplying the two given Pauli matricex
        pauli_product = np.dot(PAULI_MATRICES_DICT[pauli_i], PAULI_MATRICES_DICT[pauli_j])
        pauli_list = ["I", "X", "Y", "Z"]

        # M =  I*tr(M)/2 + X*tr(XM)/2 + Y*tr(YM)/2 + Z*tr(ZM)/2] where M is a 2x2 matrix
        coeff_lst = [np.trace(np.dot(matrix, pauli_product)) / 2 for matrix in [I2, SIGMA_X, SIGMA_Y, SIGMA_Z]]

        # returns the index of the first non-zero coefficient in the above list. 
        # note that there is only one non-zero element.
        non_zero_coeff =  coeff_lst.index(next(filter(lambda a: a != 0j, coeff_lst)))

        return coeff_lst[non_zero_coeff], pauli_list[non_zero_coeff]

r"""

This function takes the hamiltonian in terms of Pauli strings and their coefficients and multiply
it by another Pauli matrix from either left or from right and returns the result. This is used
to calculate terms in the current like L^{\dagger}_{1}HL_{1}.

"""

def hamiltonian_product_pauli(original_hamiltonian_pauli_lst, original_hamiltonian_pauli_coeff, pauli_matrix_to_multiply, action_direction):

        new_hamiltonian_pauli_lst = []
        new_hamiltonian_pauli_coeff = []        

        if action_direction == "left":

                for i in range(len(original_hamiltonian_pauli_lst)):

                        pauli_strings = original_hamiltonian_pauli_lst[i]
                        pauli_coeff  = original_hamiltonian_pauli_coeff[i]

                        """
                        The given Pauli matrix will be multiplied to the fourth qubit of each Pauli string in
                        the Hamiltonian. This is because L_1 = IIIS^{(0)}_{-}. The Pauli matrix in the original
                        string in zero th position is the Lth position.
                        """
                        operator_on_qubit_0 = pauli_strings[-1]

                        # pauli matrix multiplied from left
                        new_pauli_product = pauli_product(pauli_matrix_to_multiply,operator_on_qubit_0)

                        # updating the Pauli string
                        new_hamiltonian_pauli_lst.append(pauli_strings[0:L-1]+new_pauli_product[1])

                        # updating the Pauli coefficient
                        new_hamiltonian_pauli_coeff.append(pauli_coeff*new_pauli_product[0])

        elif action_direction == "right":

                for i in range(len(original_hamiltonian_pauli_lst)):

                        pauli_strings = original_hamiltonian_pauli_lst[i]
                        pauli_coeff  = original_hamiltonian_pauli_coeff[i]

                        """
                        The given Pauli matrix will be multiplied to the fourth qubit of each Pauli string in
                        the Hamiltonian. This is because L_1 = IIIS^{(0)}_{-}. The Pauli matrix in the original
                        string in zero th position is the Lth position.
                        """
                        operator_on_qubit_0 = pauli_strings[-1]

                        # pauli matrix multiplied from right
                        new_pauli_product = pauli_product(operator_on_qubit_0,pauli_matrix_to_multiply)

                        # updating the Pauli string
                        new_hamiltonian_pauli_lst.append(pauli_strings[0:L-1]+new_pauli_product[1])

                        # updating the Pauli coefficient
                        new_hamiltonian_pauli_coeff.append(pauli_coeff*new_pauli_product[0])

        return new_hamiltonian_pauli_lst, new_hamiltonian_pauli_coeff              

r"""
This function takes pairs each with a string of Pauli matrices and coefficents and add the
coefficients of the Pauli strings common to both the element in the pair.
For example, input: (["IXII","IZIZ"],(a,b)) and (["YYXX","IZIZ"],(c,d)) output: ([]"IXII","YYXX","IZIZ"],(a,c,b+d))
"""

def add_similar_elements(pair1, pair2):

    dict1 = dict(zip(pair1[0], pair1[1]))
    dict2 = dict(zip(pair2[0], pair2[1]))

    all_keys = set(dict1.keys()).union(set(dict2.keys()))

    result_dict = {key: dict1.get(key, 0) + dict2.get(key, 0) for key in all_keys}

    result_pair = (list(result_dict.keys()), list(result_dict.values()))

    return result_pair 

r"""
Below each term in the I_in is evaluated separately by using the functions defined above.
"""

#L_1 = III(X-1j*Y)/2
def L_1_dag_H_L_1():

        ## (1/4) (XHX - 1j*XHY + 1j*YHX + YHY)
        # XHX
        h1,p1 = hamiltonian_product_pauli(hamiltonian_pauli_lst,hamiltonian_pauli_coeff_lst,"X","left") # XH
        h2, p2 = hamiltonian_product_pauli(h1,p1,"X","right") # XHX

        # XHY
        h3,p3 = hamiltonian_product_pauli(hamiltonian_pauli_lst,hamiltonian_pauli_coeff_lst,"X","left") # XH
        h4,p4 = hamiltonian_product_pauli(h3,p3,"Y","right")  # XHY

        # YHX
        h5,p5 = hamiltonian_product_pauli(hamiltonian_pauli_lst,hamiltonian_pauli_coeff_lst,"Y","left") # YH
        h6,p6 = hamiltonian_product_pauli(h5,p5,"X","right")  # YHX

        # YHY
        h7,p7 = hamiltonian_product_pauli(hamiltonian_pauli_lst,hamiltonian_pauli_coeff_lst,"Y","left") # YH
        h8,p8 = hamiltonian_product_pauli(h7,p7,"Y","right")  # YHY

        ## adding the coefficients of similar Pauli strings
        # XHX - 1j*XHY
        h_iter_1, p_iter_1 = add_similar_elements((h2,np.array(p2)),(h4,-1j*np.array(p4)))  
        # XHX - 1j*XHY + 1j*YHX
        h_iter_2, p_iter_2 = add_similar_elements((h_iter_1,p_iter_1),(h6,1j*np.array(p6)))
        # XHX - 1j*XHY + 1j*YHX + YHY
        h_iter_3, p_iter_3 = add_similar_elements((h_iter_2,p_iter_2),(h8,np.array(p8)))

        return h_iter_3, (1/4)*np.array(p_iter_3)


def H_L1_dag_L1():

        ## H L1_dag L_1 = (1/2)*(HI + HZ)

        # H*I
        h1,p1 = hamiltonian_product_pauli(hamiltonian_pauli_lst,hamiltonian_pauli_coeff_lst,"I","right")
        # HZ
        h2,p2 = hamiltonian_product_pauli(hamiltonian_pauli_lst,hamiltonian_pauli_coeff_lst,"Z","right")

        h_iter_1, p_iter_1 = add_similar_elements((h1,np.array(p1)),(h2,np.array(p2)))

        return h_iter_1, (1/2)*np.array(p_iter_1)


def L_1_dag_L_1_H():

        # L1_dag L_1 H = (1/2)*(H + ZH)

        # I*H
        h1,p1 = hamiltonian_product_pauli(hamiltonian_pauli_lst,hamiltonian_pauli_coeff_lst,"I","left")
        # ZH
        h2,p2 = hamiltonian_product_pauli(hamiltonian_pauli_lst,hamiltonian_pauli_coeff_lst,"Z","left")

        h_iter_1, p_iter_1 = add_similar_elements((h1,np.array(p1)),(h2,np.array(p2)))

        return h_iter_1, (1/2)*np.array(p_iter_1)

# L_1_dag_H_L_1 - (1/2)*H_L1_dag_L1
I_in_term_1_term_2 = add_similar_elements((L_1_dag_H_L_1()[0],L_1_dag_H_L_1()[1]),(H_L1_dag_L1()[0],(-1/2)*np.array(H_L1_dag_L1()[1])))

# L_1_dag_H_L_1 - (1/2)*H_L1_dag_L1 - (1/2)*L_1_dag_L_1_H
I_in_term_1_term_2_term_3 = add_similar_elements((I_in_term_1_term_2[0],I_in_term_1_term_2[1]),(L_1_dag_L_1_H()[0],(-1/2)*np.array(L_1_dag_L_1_H()[1])))


#L_2 = III(X+1j*Y)/2
def L_2_dag_H_L_2():

        ## (1/4) (XHX + 1j*XHY - 1j*YHX + YHY)
        # XHX
        h1,p1 = hamiltonian_product_pauli(hamiltonian_pauli_lst,hamiltonian_pauli_coeff_lst,"X","left") # XH
        h2, p2 = hamiltonian_product_pauli(h1,p1,"X","right") # XHX

        # XHY
        h3,p3 = hamiltonian_product_pauli(hamiltonian_pauli_lst,hamiltonian_pauli_coeff_lst,"X","left") # XH
        h4,p4 = hamiltonian_product_pauli(h3,p3,"Y","right")  # XHY

        # YHX
        h5,p5 = hamiltonian_product_pauli(hamiltonian_pauli_lst,hamiltonian_pauli_coeff_lst,"Y","left") # YH
        h6,p6 = hamiltonian_product_pauli(h5,p5,"X","right")  # YHX

        # YHY
        h7,p7 = hamiltonian_product_pauli(hamiltonian_pauli_lst,hamiltonian_pauli_coeff_lst,"Y","left") # YH
        h8,p8 = hamiltonian_product_pauli(h7,p7,"Y","right")  # YHY

        ## adding the coefficients of similar Pauli strings
        # XHX + 1j*XHY
        h_iter_1, p_iter_1 = add_similar_elements((h2,np.array(p2)),(h4,1j*np.array(p4)))  
        # XHX + 1j*XHY - 1j*YHX
        h_iter_2, p_iter_2 = add_similar_elements((h_iter_1,p_iter_1),(h6,-1j*np.array(p6)))
        # XHX + 1j*XHY - 1j*YHX + YHY
        h_iter_3, p_iter_3 = add_similar_elements((h_iter_2,p_iter_2),(h8,np.array(p8)))

        return h_iter_3, (1/4)*np.array(p_iter_3)


def H_L2_dag_L2():

        ## H L2_dag L_2 = (1/2)*(HI - HZ)

        # HI
        h1,p1 = hamiltonian_product_pauli(hamiltonian_pauli_lst,hamiltonian_pauli_coeff_lst,"I","right")
        # HZ
        h2,p2 = hamiltonian_product_pauli(hamiltonian_pauli_lst,hamiltonian_pauli_coeff_lst,"Z","right")

        h_iter_1, p_iter_1 = add_similar_elements((h1,np.array(p1)),(h2,-np.array(p2)))

        return h_iter_1, (1/2)*np.array(p_iter_1)


def L_2_dag_L_2_H():

        # L2_dag L_2 H = (1/2)*(IH - ZH)

        # IH
        h1,p1 = hamiltonian_product_pauli(hamiltonian_pauli_lst,hamiltonian_pauli_coeff_lst,"I","left")
        # ZH
        h2,p2 = hamiltonian_product_pauli(hamiltonian_pauli_lst,hamiltonian_pauli_coeff_lst,"Z","left")
        
        h_iter_1, p_iter_1 = add_similar_elements((h1,np.array(p1)),(h2,-np.array(p2)))

        return h_iter_1, (1/2)*np.array(p_iter_1)


# L_2_dag_H_L_2 - (1/2)*H_L2_dag_L2
I_out_term_1_term_2 = add_similar_elements((L_2_dag_H_L_2()[0],L_2_dag_H_L_2()[1]),(H_L2_dag_L2()[0],(-1/2)*np.array(H_L2_dag_L2()[1])))

# L_2_dag_H_L_2 - (1/2)*H_L2_dag_L2 - (1/2)*L_2_dag_L_2_H
I_out_term_1_term_2_term_3 = add_similar_elements((I_out_term_1_term_2[0],I_out_term_1_term_2[1]),(L_2_dag_L_2_H()[0],(-1/2)*np.array(L_2_dag_L_2_H()[1])))

# %%
r"""
Pauli strings whose corresponding coefficient is zero need not be measured. So they will be removed.
"""

# Unpack the pair into two lists
list1, list2 = I_in_term_1_term_2_term_3

# Use a list comprehension to filter the elements where the corresponding element in the second list is not 0 or 0j
filtered_I_in_term_1_term_2_term_3 = ([x for x, y in zip(list1, list2) if y != 0j], [y for y in list2 if y != 0j])

# %%
# Unpack the pair into two lists
list1, list2 = I_out_term_1_term_2_term_3

# Use a list comprehension to filter the elements where the corresponding element in the second list is not 0 or 0j
filtered_I_out_term_1_term_2_term_3 = ([x for x, y in zip(list1, list2) if y != 0j], [y for y in list2 if y != 0j])

# %% [markdown]
# #### Noise model for the simulator

# %%
r"""
This part of the code builds the noise model for the Qiskit simulator. We only consider T_{1} and T_{2}
noise. This part of the code is taken from Qiskit website https://qiskit.org/ecosystem/aer/tutorials/3_building_noise_models.html
"""

noise_index = int(sys.argv[1])

noise_factor = np.linspace(1,300,16)

T1_noise = 213.07e3/noise_factor[noise_index]
T2_noise = 115.57e3/noise_factor[noise_index]

T1_standard_deviation = T1_noise/4
T2_standard_deviation = T2_noise/4
# T1 and T2 values for qubits
T1s = np.random.normal(T1_noise, T1_standard_deviation, L+1) # Sampled from normal distribution mean T_{1} microsec
T2s = np.random.normal(T2_noise, T2_standard_deviation, L+1)  # Sampled from normal distribution mean T_{2} microsec

# Truncate random T2s <= T1s
T2s = np.array([min(T2s[j], 2 * T1s[j]) for j in range(L+1)])

r"""
If set the time below to zero then the simulator is ideal.
"""
# Instruction times (in nanoseconds)
time_u1         = 0   # virtual gate
time_u2         = 50  # (single X90 pulse)
time_u3         = 100 # (two X90 pulses)
time_sx         = 100
time_cx         = 300
time_reset      = 1000  # 1 microsecond
time_measure    = 1000 # 1 microsecond

# QuantumError objects
errors_reset = [thermal_relaxation_error(t1, t2, time_reset)
                for t1, t2 in zip(T1s, T2s)]
errors_measure = [thermal_relaxation_error(t1, t2, time_measure)
                  for t1, t2 in zip(T1s, T2s)]
errors_u1  = [thermal_relaxation_error(t1, t2, time_u1)
              for t1, t2 in zip(T1s, T2s)]
errors_u2  = [thermal_relaxation_error(t1, t2, time_u2)
              for t1, t2 in zip(T1s, T2s)]
errors_u3  = [thermal_relaxation_error(t1, t2, time_u3)
              for t1, t2 in zip(T1s, T2s)]
errors_sx  = [thermal_relaxation_error(t1, t2, time_sx)
              for t1, t2 in zip(T1s, T2s)]              
errors_cx = [[thermal_relaxation_error(t1a, t2a, time_cx).expand(
             thermal_relaxation_error(t1b, t2b, time_cx))
              for t1a, t2a in zip(T1s, T2s)]
               for t1b, t2b in zip(T1s, T2s)]
errors_ecr = [[thermal_relaxation_error(t1a, t2a, time_cx).expand(
             thermal_relaxation_error(t1b, t2b, time_cx))
              for t1a, t2a in zip(T1s, T2s)]
               for t1b, t2b in zip(T1s, T2s)]               

# Add errors to noise model
noise_thermal = NoiseModel()
for j in range(L+1):
    noise_thermal.add_quantum_error(errors_reset[j], "reset", [j])
    noise_thermal.add_quantum_error(errors_measure[j], "measure", [j])
    noise_thermal.add_quantum_error(errors_sx[j], "sx",[j])
    noise_thermal.add_quantum_error(errors_sx[j], "rz",[j])
    #noise_thermal.add_quantum_error(errors_u1[j], "u1", [j])
    #noise_thermal.add_quantum_error(errors_u2[j], "u2", [j])
    #noise_thermal.add_quantum_error(errors_u3[j], "u3", [j])
    for k in range(L+1):
        noise_thermal.add_quantum_error(errors_cx[j][k], "cx", [j, k])
        noise_thermal.add_quantum_error(errors_ecr[j][k], "ecr", [j, k])   
        
#print(noise_thermal)    

# %% [markdown]
# #### Basis change circuit for $X$ and $Y$ measurement

# %%
r"""
After preparing the density matrix using the quantum circuit, Pauli strings with only "I" and "Z"
can be measured directly. To measure Pauli strings involving "X" and "Y" one has to do a basis
transformation before measurement. This can be done by using the identities HZH = X and S^{\dagger}HZHS = Y.
This function below appends a H gate for X measurement and S_{\dagger},H gate for Y measurement.
"""

def pauli_string_basis_change_circuit(observable_pauli_string):
    
    # existing quantum registers
    qr = QuantumRegister(L,"q")
    anc = QuantumRegister(1,"ancilla")

    # add a classical register
    cr = ClassicalRegister(L,"c")
    
    # create a new quantum circuit with the classical register
    qc_basis_change = QuantumCircuit(anc,qr, cr)

    # reversing the pauli string so that the rightmost qubit is counted as zeroth qubit
    observable_pauli_string = observable_pauli_string[::-1]
    
    for i in range(L):

        if observable_pauli_string[i] == "I":
            pass
        elif observable_pauli_string[i] == "Z":
            pass
        elif observable_pauli_string[i] == "X":
            qc_basis_change.h(qr[i])
        elif observable_pauli_string[i] == "Y":
            qc_basis_change.sdg(qr[i])
            qc_basis_change.h(qr[i])

    # measure all qubits
    for i in range(L):
        qc_basis_change.measure(qr[i], cr[i])   

    return qc_basis_change

#pauli_string_1 = filtered_I_out_term_1_term_2_term_3[0][3]
#print(pauli_string_1)    
#pauli_string_basis_change_circuit(pauli_string_1).draw("mpl", style= "iqp",scale=2)

# %%
r"""
This function takes a Pauli string as input and returns the counts from its simulation.
"""

def trotter_simulation_and_return_counts(pauli_string_to_calculate_expectation_value,
                                         time_step_for_trotterization,
                                         initial_state_of_system,
                                         t_final,
                                         number_of_shots,
                                         ):
        
        # constructs the trotter circuit and prepares the density matrix
        trotter_circuit = time_evolved_density_matrix(time_step_for_trotterization,
                                                      t_final,
                                                      initial_state_of_system)

        # composes necessary basis change to measures pauli operators X and Y if exists in pauli string
        trotter_circuit = trotter_circuit.compose(pauli_string_basis_change_circuit(pauli_string_to_calculate_expectation_value))

        sim_thermal = AerSimulator(noise_model=noise_thermal) # noisy simulator
        #sim_thermal = Aer.get_backend("qasm_simulator") # ideal simulator

        basis_set_1 = ["cx","rz","h"]
        basis_set_2 = ["ecr","id","rz","sx","x"]
        transpiled_trotter_circuit = transpile(trotter_circuit, sim_thermal, basis_gates = basis_set_1 ,optimization_level = 2)

        result_thermal = sim_thermal.run(transpiled_trotter_circuit, shots = number_of_shots).result()

        counts_thermal = result_thermal.get_counts()

        return counts_thermal

# %%
r"""
This function calculates the expectation value of each pauli string in the current in and current out operator.
Input: List of Pauli strings to calculate expectation value 
e.g. ["XXYY", "IZIZ"]
Output: List of expectation values of each Pauli string in the input list
e.g. [Expectation value("XXYY"), Expectation value("IZIZ")]
"""

def expectation_value_of_pauli_string_from_counts(pauli_string, counts_dictionary_from_measurement):

        pauli_string_expectation_value = 0.0
        non_identity_indices = [i for i, char in enumerate(pauli_string) if char != "I"]
        for key, value in counts_dictionary_from_measurement.items():
                filtered_binary_string = "".join([char for i, char in enumerate(key) if i in non_identity_indices])
                pauli_string_expectation_value += ((-1)**sum(int(num) for num in filtered_binary_string))* (value/NUMBER_OF_SHOTS)
        return pauli_string_expectation_value

def current_expectation_value(current_operator_pauli_strings, time):

        current_expectation_value_lst = []

        for pauli_string in current_operator_pauli_strings:

                # simulating a circuit with the Pauli string
                bit_strings_and_counts = trotter_simulation_and_return_counts(pauli_string,
                                                0.1,"0"*NUMBER_OF_QUBITS,time,NUMBER_OF_SHOTS)


                # calculating the expectation value of the Pauli string from counts       
                pauli_string_expectation_value = expectation_value_of_pauli_string_from_counts(pauli_string, bit_strings_and_counts)

                current_expectation_value_lst.append(pauli_string_expectation_value)                                

        return current_expectation_value_lst

# %%
I_in_pauli_list = filtered_I_in_term_1_term_2_term_3[0]
I_in_coefficients = filtered_I_in_term_1_term_2_term_3[1]

I_out_pauli_list = filtered_I_out_term_1_term_2_term_3[0]
I_out_coefficients = filtered_I_out_term_1_term_2_term_3[1]

time_lst = np.linspace(0.0,2,5)

for time in time_lst: 
        t = current_expectation_value(I_in_pauli_list, time)
        np.save("t_"+str(np.around(time,2))+".npy",t)


