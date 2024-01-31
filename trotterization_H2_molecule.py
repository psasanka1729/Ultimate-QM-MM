import sys
import numpy as np
from qiskit import*
import qiskit
from qiskit import transpile,QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit import Aer
from qiskit.quantum_info import partial_trace
# Import from Qiskit Aer noise module
from qiskit_aer.noise import (NoiseModel, QuantumError, ReadoutError,
    pauli_error, depolarizing_error, thermal_relaxation_error)
from qiskit.quantum_info import DensityMatrix
from scipy.sparse import csr_matrix, kron

def partial_trace_4_4(matrix):
        return np.matrix([[matrix[0,0]+matrix[1,1], matrix[0,2]+matrix[1,3]],
                          [matrix[0,2]+matrix[3,1], matrix[2,2]+matrix[3,3]]])

I2 = np.array([[1,0],[0,1]])
sigma_x =np.array([[0,1],[1,0]])
sigma_y = np.array([[0,-1j],[1j,0]])
sigma_z = np.array([[1,0],[0,-1]])

PI_0 = np.matrix([[1,0],[0,0]])
PI_1 = np.matrix([[0,0],[0,1]])                          

def controlled_ry_gate_matrix(angle_theta):

        def ry_matrix(angle):
                return np.matrix([[np.cos(angle/2),-np.sin(angle/2)],
                                  [np.sin(angle/2),np.cos(angle/2)]])
        return kron(PI_0,I2) + kron(PI_1,ry_matrix(angle_theta))

def controlled_not_gate_matrix():
        return kron(I2,PI_0) + kron(sigma_x,PI_1)

# H2 molecule after Jordan Wigner transformation
L = 4
H_pauli_lst = ['IIII','IIIZ', 'IIZI', 'IIZZ', 'IZII',
               'IZIZ', 'ZIII', 'ZIIZ','YYYY', 'XXYY', 'YYXX', 'XXXX','IZZI', 'ZIZI', 'ZZII']
H_pauli_coeff_lst = [-0.81054798+0.j,  0.17218393+0.j, -0.22575349+0.j,  0.12091263+0.j,
  0.17218393+0.j,  0.16892754+0.j, -0.22575349+0.j,  0.16614543+0.j,
  0.0452328 +0.j,  0.0452328 +0.j,  0.0452328 +0.j,  0.0452328 +0.j,
  0.16614543+0.j,  0.17464343+0.j,  0.12091263+0.j]


def sparse_Pauli_to_dense_matrix(sparse_Pauli_matrices_lst, Pauli_matrices_coefficients):
    
    # Pauli matrices dictionary
    pauli_matrices_dict = {"I": np.array([[1,0],
                                          [0,1]]),
                  "X": np.array([[0,1],
                                 [1,0]]),
                  "Y": np.array([[0,-1j],
                                 [1j,0]]),
                  "Z": np.array([[1,0],
                                 [0,-1]])}
    
    sparse_pauli_matrices_dict = {key: csr_matrix(value) for key, value in pauli_matrices_dict.items()}      

    sparse_hamiltonian = csr_matrix((2**L, 2**L))

    # converts a pauli string into a matrix
    def pauli_string_to_matrix(pauli_string):
        mat = sparse_pauli_matrices_dict[pauli_string[0]]
        """for p_string in range(L-2,-1,-1):
          mat = kron(sparse_pauli_matrices_dict[pauli_string[p_string]],mat)
        return mat"""
        for p_string in range(1,L):
           mat = kron(mat,sparse_pauli_matrices_dict[pauli_string[p_string]])
        return mat
    
    for i in range(len(sparse_Pauli_matrices_lst)):
      
      sparse_hamiltonian += Pauli_matrices_coefficients[i] * pauli_string_to_matrix(sparse_Pauli_matrices_lst[i])
      
    return sparse_hamiltonian

full_hamiltonian = sparse_Pauli_to_dense_matrix(H_pauli_lst,H_pauli_coeff_lst);

gamma_in = 0.6
gamma_out = 0.6

# %% [markdown]
# ### $n=1$ and $n=2$ states of the Hamiltonian

# %%
"""# n = 1 basis states 0001,0010,0100,1000
def ket_from_binary_string(binary_string):
        bin_dict = {"0":np.matrix([1,0]), "1":np.matrix([0,1])}
        b_0 = bin_dict[binary_string[-1]]
        for b in range(len(binary_string)-2,-1,-1):
                b_0 = kron(bin_dict[binary_string[b]],b_0)
        return b_0

n_1_sector_basis_states = [ket_from_binary_string("0001"),ket_from_binary_string("0010"),ket_from_binary_string("0100"),
                           ket_from_binary_string("1000")]

n_1_sector_size = 4
H_n_1_sector = np.zeros((n_1_sector_size,n_1_sector_size),dtype=np.complex128)

for m in range(len(n_1_sector_basis_states)):
        for n in range(len(n_1_sector_basis_states)):
                ket_m = n_1_sector_basis_states[m]
                ket_n = n_1_sector_basis_states[n]
                H_n_1_sector[m,n] = (ket_m *full_hamiltonian* ket_n.T).A[0,0]


# n = 2 basis states 0011,0101,0110,1001,1010,1100
n_2_sector_basis_states = [ket_from_binary_string("0011"),ket_from_binary_string("0101"),ket_from_binary_string("0110"),
                           ket_from_binary_string("1001"),ket_from_binary_string("1010"),
                           ket_from_binary_string("1100")]

n_2_sector_size = 6
H_n_2_sector = np.zeros((n_2_sector_size,n_2_sector_size),dtype=np.complex128)

for m in range(len(n_2_sector_basis_states)):
        for n in range(len(n_2_sector_basis_states)):
                ket_m = n_2_sector_basis_states[m]
                ket_n = n_2_sector_basis_states[n]
                H_n_2_sector[m,n] = (ket_m *full_hamiltonian* ket_n.T).A[0,0]""";

# %%
#L = 4
def one_time_step_circuit(dt,barrier_status):

    def ZZ_gate_circuit(qubit_2, qubit_1, coefficient, delta_t): # count qubits from right as in Qiskit
        qc_zz = QuantumCircuit(L)
        qc_zz.cx(qubit_1,qubit_2)
        qc_zz.rz(2*coefficient*delta_t, qubit_2)
        qc_zz.cx(qubit_1,qubit_2)
        return qc_zz
    
    # Existing quantum registers
    qr = QuantumRegister(L,"q")
    anc = QuantumRegister(1,r"\rm{ancilla}")


    # Create a new quantum circuit with the classical register
    qc_h2 = QuantumCircuit(anc,qr)    
    #qc_h2 = QuantumCircuit(L+1)

    # one Z gate 
    # 1,2,4,6
    qc_h2.rz(2*H_pauli_coeff_lst[1].real*dt,0)
    qc_h2.rz(2*H_pauli_coeff_lst[2].real*dt,1)
    qc_h2.rz(2*H_pauli_coeff_lst[4].real*dt,2)
    qc_h2.rz(2*H_pauli_coeff_lst[6].real*dt,3)
    if barrier_status == True:
        qc_h2.barrier()
    else:
        pass

    # two z gates
    # 3,5,7,12,13,14
    # IIZZ 3
    qc_h2 = qc_h2.compose(ZZ_gate_circuit(1,0,H_pauli_coeff_lst[3].real,dt))
    # ZZII 14
    qc_h2 = qc_h2.compose(ZZ_gate_circuit(3,2,H_pauli_coeff_lst[14].real,dt))     
    if barrier_status == True:
        qc_h2.barrier()
    else:
        pass

    # IZIZ 5
    qc_h2.swap(1,2)
    qc_h2 = qc_h2.compose(ZZ_gate_circuit(1,0,H_pauli_coeff_lst[5].real,dt))
    qc_h2.swap(1,2)    
    if barrier_status == True:
        qc_h2.barrier()
    else:
        pass

    # ZIZI 13
    qc_h2.swap(1,2)
    qc_h2 = qc_h2.compose(ZZ_gate_circuit(3,2,H_pauli_coeff_lst[13].real,dt))   
    qc_h2.swap(1,2)
    if barrier_status == True:
        qc_h2.barrier()
    else:
        pass

    # ZIIZ 7 
    qc_h2.swap(3,2)
    qc_h2.swap(2,1)        
    qc_h2 = qc_h2.compose(ZZ_gate_circuit(1,0,H_pauli_coeff_lst[7].real,dt)) 
    qc_h2.swap(2,1)       
    qc_h2.swap(3,2)        
    if barrier_status == True:
        qc_h2.barrier()
    else:
        pass
    # IZZI 12
    qc_h2 = qc_h2.compose(ZZ_gate_circuit(2,1,H_pauli_coeff_lst[12].real,dt))    

    if barrier_status == True:
        qc_h2.barrier()
    else:
        pass

    # XXXX 
    for i in range(L):
        qc_h2.h(i)

    qc_h2.cx(0,1)
    qc_h2.cx(1,2)
    qc_h2.cx(2,3)    
    qc_h2.rz(2*H_pauli_coeff_lst[11].real*dt,3)
    qc_h2.cx(2,3)
    qc_h2.cx(1,2)
    qc_h2.cx(0,1)

    for i in range(L):
        qc_h2.h(i)   
    if barrier_status == True:
        qc_h2.barrier()
    else:
        pass   

    # YYYY
    for i in range(L):
        qc_h2.sdg(i)    
    for i in range(L):
        qc_h2.h(i)       
    qc_h2.cx(0,1)
    qc_h2.cx(1,2)
    qc_h2.cx(2,3)    
    qc_h2.rz(2*H_pauli_coeff_lst[8].real*dt,3)
    qc_h2.cx(2,3)
    qc_h2.cx(1,2)
    qc_h2.cx(0,1)
    for i in range(L):
        qc_h2.h(i)  
    for i in range(L):
        qc_h2.sdg(i)  
    if barrier_status == True:
        qc_h2.barrier()
    else:
        pass   

    # XXYY
    qc_h2.sdg(0)
    qc_h2.h(0)
    qc_h2.sdg(1)
    qc_h2.h(1)   
    qc_h2.h(2)
    qc_h2.h(3)

    qc_h2.cx(0,1)
    qc_h2.cx(1,2)
    qc_h2.cx(2,3)    
    qc_h2.rz(2*H_pauli_coeff_lst[9].real*dt,3)
    qc_h2.cx(2,3)
    qc_h2.cx(1,2)
    qc_h2.cx(0,1)   

    qc_h2.h(0)
    qc_h2.sdg(0)
    qc_h2.h(1) 
    qc_h2.sdg(1)  
    qc_h2.h(2)
    qc_h2.h(3)     
    if barrier_status == True:
        qc_h2.barrier()
    else:
        pass

    # YYXX
    qc_h2.sdg(2)
    qc_h2.h(2)
    qc_h2.sdg(3)
    qc_h2.h(3)   
    qc_h2.h(0)
    qc_h2.h(1)    
    qc_h2.cx(0,1)
    qc_h2.cx(1,2)
    qc_h2.cx(2,3)    
    qc_h2.rz(2*H_pauli_coeff_lst[10].real*dt,3)
    qc_h2.cx(2,3)
    qc_h2.cx(1,2)
    qc_h2.cx(0,1)
    qc_h2.h(2)  
    qc_h2.sdg(2)
    qc_h2.h(3)   
    qc_h2.sdg(3)
    qc_h2.h(0)
    qc_h2.h(1)

    #qc_h2.barrier()

    # ancilla qubit
    # L_out
    theta_out = 2*np.arcsin(np.sqrt(dt*gamma_out))
    qc_h2.initialize([1,0],anc[0])
    qc_h2.cry(theta_out,qr[0],anc)
    qc_h2.cx(anc,qr[0])
    qc_h2.initialize([1,0],anc)    
    if barrier_status == True:
        qc_h2.barrier()
    else:
        pass        
    # L_in
    qc_h2.x(qr[0])
    theta_in = 2*np.arcsin(np.sqrt(dt*gamma_in))
    qc_h2.cry(theta_in,qr[0],anc)
    qc_h2.cx(anc,qr[0])
    qc_h2.x(qr[0])
    qc_h2.initialize([1,0],anc)
    return qc_h2


def trotter_circuit(time_step,final_time,initial_state):
    number_of_iterations = int(final_time/time_step)
    print("Number of Floquet cycles = ", number_of_iterations)
    transpiled_one_step_circuit = one_time_step_circuit(time_step,False)
    #transpiled_one_step_circuit = transpile(one_time_step_circuit(time_step,False), basis_gates = ["rz","cx","h",], optimization_level=2)

    # Existing quantum registers
    qr = QuantumRegister(L,"q")
    anc = QuantumRegister(1,"ancilla")

    # Add a classical register
    cr = ClassicalRegister(L+1,"c")
    # Create a new quantum circuit with the classical register
    qc = QuantumCircuit(anc,qr, cr)

    # system is initialized at the user defined state
    qubit_dict = {"1":[0,1],"0":[1,0]}

    r"""
        The initial state is assigned in reverse order because of qiskit's convention
        of counting from right to left. So the first value of the initial state string
        is assigned to the bottom qubit and the last value of the initial state string
        is assigned to the top qubit!
    """
    qc.initialize(qubit_dict[initial_state[3]],qr[0])
    qc.initialize(qubit_dict[initial_state[2]],qr[1])
    qc.initialize(qubit_dict[initial_state[1]],qr[2])
    qc.initialize(qubit_dict[initial_state[0]],qr[3])

    for _ in range(number_of_iterations):
        qc = qc.compose(transpiled_one_step_circuit)
    qc.measure(anc,cr[4])
    qc.measure(qr[3], cr[3])   
    qc.measure(qr[2], cr[2])
    qc.measure(qr[1], cr[1])
    qc.measure(qr[0], cr[0])

    #print("Circuit depth = ",qc.depth())
    return qc


#T1_noise_lst = [10e3,20e3,40e3,60e3,80e3,100e3,200e3,400e3,600e3,800e3]
#T2_noise_lst = [10e3,20e3,40e3,60e3,80e3,100e3,200e3,400e3,600e3,800e3]

noise_index = int(sys.argv[1])

noise_factor = np.linspace(1,3,16)
#T1_noise = T1_noise_lst[noise_index]
#T2_noise = T2_noise_lst[noise_index]

T1_noise = 213.07e3*noise_factor
T2_noise = 115.57e3*noise_factor

T1_standard_deviation = T1_noise/4
T2_standard_deviation = T2_noise/4

# T1 and T2 values for qubits 0-3
T1s = np.random.normal(T1_noise, T1_standard_deviation, L+1) # Sampled from normal distribution mean 50 microsec
T2s = np.random.normal(T2_noise, T2_standard_deviation, L+1)  # Sampled from normal distribution mean 50 microsec

# Truncate random T2s <= T1s
T2s = np.array([min(T2s[j], 2 * T1s[j]) for j in range(L+1)])

# Instruction times (in nanoseconds)
time_u1 = 0   # virtual gate
time_u2 = 50  # (single X90 pulse)
time_u3 = 100 # (two X90 pulses)
time_sx = 100
time_cx = 300
time_reset = 1000  # 1 microsecond
time_measure = 1000 # 1 microsecond

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
    #noise_thermal.add_quantum_error(errors_u1[j], "u1", [j])
    #noise_thermal.add_quantum_error(errors_u2[j], "u2", [j])
    #noise_thermal.add_quantum_error(errors_u3[j], "u3", [j])
    for k in range(L+1):
        noise_thermal.add_quantum_error(errors_cx[j][k], "cx", [j, k])
    for k in range(L+1):
        noise_thermal.add_quantum_error(errors_ecr[j][k], "ecr", [j, k])              

def counts_to_statevector(counts):
    # Get the total number of counts
    total_counts = sum(counts.values())
    
    # Initialize an empty statevector
    statevector = []
    
    # For each state in the counts
    for state, count in counts.items():
        # Calculate the probability of the state
        probability = count / total_counts
        
        # Convert the binary state to a decimal index
        index = int(state, 2)
        
        # Initialize a state with all zeros
        current_state = np.zeros((2**len(state),), dtype=np.complex128)
        
        # Set the amplitude of the state to the square root of the probability
        current_state[index] = np.sqrt(probability)
        
        # Add the state to the statevector
        statevector.append(current_state)
    
    # Return the sum of all states
    return sum(statevector)

time_step_for_trotterization = 0.1
time_lst = np.linspace(time_step_for_trotterization,100,50)
counts_lst = []
density_matrices_lst = []
initial_state_of_system = "0100"

for time in time_lst:
        # Execute and get counts
        h_2_molecule_circuit = trotter_circuit(time_step_for_trotterization,time,initial_state_of_system)
        # Run the noisy simulation
        sim_thermal = AerSimulator(noise_model=noise_thermal_single_qubit)
        circ_tthermal = transpile(h_2_molecule_circuit, sim_thermal, basis_gates = ["ecr","id","rz","sx","reset","x"], optimization_level = 2)

        #density_matrix = DensityMatrix.from_instruction(circ_tthermal)

        result_thermal = sim_thermal.run(circ_tthermal).result()
        #statevector = result.get_statevector()
        counts_thermal = result_thermal.get_counts()
        statevector = counts_to_statevector(counts_thermal)
        density_matrix = np.outer(statevector, np.conj(statevector))  
        #print(statevector.size)
        reduced_density_matrix = partial_trace(density_matrix, [0])
        density_matrices_lst.append(reduced_density_matrix)

I2 = np.matrix([[1,0],
                [0,1]])
sigma_x = np.matrix([[0,1],
                     [1,0]])
sigma_y = np.matrix([[0,-1j],
                     [1j,0]])
sigma_z = np.matrix([[1,0],
                     [0,-1]])

# sigma_- acting on the first qubit
L_in  = kron(I2,kron(I2,kron(I2,(sigma_x - 1j*sigma_y)/2)))
L_out = kron(I2,kron(I2,kron(I2,(sigma_x + 1j*sigma_y)/2)))

# energy currents definition
def I_in(hamiltonian, density_matrix):
        return gamma_in*np.trace((L_in.conj().T@hamiltonian@L_in
                - (1/2)*hamiltonian@L_in.conj().T@L_in
                - (1/2)*L_in.conj().T@L_in@hamiltonian)@density_matrix)

def I_out(hamiltonian, density_matrix):
        return gamma_out*np.trace((L_out.conj().T@hamiltonian@L_out
                - (1/2)*hamiltonian@L_out.conj().T@L_out
                - (1/2)*L_out.conj().T@L_out@hamiltonian)@density_matrix)

# %%
I_in_lst = []
I_out_lst = []
for rho in density_matrices_lst:
        I_in_lst.append(I_in(full_hamiltonian,rho).real)
        I_out_lst.append(I_out(full_hamiltonian,rho).real)

np.save("density_matrices_lst.npy",density_matrices_lst)
np.save("I_in_lst.npy",I_in_lst)
np.save("I_out_lst.npy",I_out_lst)
