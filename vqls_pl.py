import pennylane as qml
from pennylane import numpy as np
from braket.jobs.metrics import log_metric
from braket.jobs import save_job_result
from braket.jobs.environment_variables import get_hyperparameters, get_job_device_arn, get_job_name
from braket.tracking import Tracker

print(f"Starting job {get_job_name()}")
task_tracker = Tracker().start()
circ_invoc = 0

# Let's solve the following 2^10 times 2^10 system:
#
# A = 1.00 * I_{\otimes 10} + 0.25 Z_2 \otimes X_5 \otimes X_9 + 0.25 * X_0 \otimes Z_1 \otimes X_4 \otimes X_8
# b = H_0 \otimes H_1 \otimes H_2 \otimes H_3 \otimes H_6 \otimes H_7 |0>
#
n_qubits = 10  # Number of system qubits
tot_qubits = n_qubits + 1  # Addition of an ancillary qubit
ancilla_idx = n_qubits  # Index of the ancillary qubit (last position)
c_l = np.array([1.0, 0.25, 0.25]) # coefficients of the linear combination of A = c_0 * A_0 + c_1 * A_1 + c_2 * A_2

# Load hyperparameters for training
hyper_params = get_hyperparameters()
print(f"Hyper parameters: {hyper_params}")
steps = int(hyper_params['n_steps'])          # number of interations for training
step_size = float(hyper_params['step_size'])  # learning rate of gradient-descend optimizer
q_delta = float(hyper_params['q_delta'])      # spread of initial random value for variational parameters
rng_seed = 0                                  # seed for the random number generator

# Set up the quantum device
n_shots = None if hyper_params['n_shots'] == 'None' else int(hyper_params['n_shots']) # Number of quantum measurements
device_for_training = qml.device("lightning.qubit", wires=tot_qubits, shots=n_shots)
device_for_sampling = qml.device("lightning.qubit", wires=n_qubits, shots=n_shots)
print(device_for_training)

def U_b():
    """Unitary matrix rotating the ground state to the problem vector |b> = U_b |0>."""
    for idx in [0, 1, 2, 3, 6, 7]:
        qml.Hadamard(wires=idx)
        
def ctrl_A_l(l):
    """Controlled versions of the unitary components A_l of the problem matrix A."""
    if l == 0: # A_0 = I_{\otimes 10}
        # We can omit qml.crtl(qml.I, control=(ancilla_idx)) operations
        None

    elif l == 1: # A_1 = Z_2 \otimes X_5 \otimes X_9
        qml.ctrl(qml.Z, control=(ancilla_idx))(wires=2)
        qml.ctrl(qml.X, control=(ancilla_idx))(wires=5)
        qml.ctrl(qml.X, control=(ancilla_idx))(wires=9)

    elif l == 2: # A_2 = X_0 \otimes Z_1 \otimes X_4 \otimes X_8
        qml.ctrl(qml.X, control=(ancilla_idx))(wires=0)
        qml.ctrl(qml.Z, control=(ancilla_idx))(wires=1)
        qml.ctrl(qml.X, control=(ancilla_idx))(wires=4)
        qml.ctrl(qml.X, control=(ancilla_idx))(wires=8)

def V(alphas):
    """Variational circuit mapping the ground state |0> to the ansatz state |x>."""
    # We first prepare an equal superposition of all the states of the computational basis.
    for idx in range(n_qubits):
        qml.Hadamard(wires=idx)

    # A very minimal variational circuit.
    for idx, element in enumerate(alphas):
        qml.RY(element, wires=idx)

@qml.qnode(device_for_training, interface="autograd")
def local_hadamard_test(alphas, l=None, lp=None, j=None, part=None):
    global circ_invoc
    circ_invoc += 1

    # First Hadamard gate applied to the ancillary qubit.
    qml.Hadamard(wires=ancilla_idx)

    # For estimating the imaginary part of the coefficient "mu", we must add a "-i"
    # phase gate.
    if part == "Im":
        qml.PhaseShift(-np.pi / 2, wires=ancilla_idx)

    # Variational circuit generating a guess for the solution vector |x>
    V(alphas)

    # Controlled application of the unitary component A_l of the problem matrix A.
    ctrl_A_l(l)

    # Adjoint of the unitary U_b associated to the problem vector |b>.
    # In this specific example Adjoint(U_b) = U_b.
    U_b()

    # Controlled Z operator at position j. If j = -1, apply the identity.
    if j != -1:
        qml.CZ(wires=[ancilla_idx, j])

    # Unitary U_b associated to the problem vector |b>.
    U_b()

    # Controlled application of Adjoint(A_lp).
    # In this specific example Adjoint(A_lp) = A_lp.
    ctrl_A_l(lp)

    # Second Hadamard gate applied to the ancillary qubit.
    qml.Hadamard(wires=ancilla_idx)

    # Expectation value of Z for the ancillary qubit.
    return qml.expval(qml.PauliZ(wires=ancilla_idx))

def mu(alphas, l=None, lp=None, j=None):
    """Generates the coefficients to compute the "local" cost function C_L."""

    mu_real = local_hadamard_test(alphas, l=l, lp=lp, j=j, part="Re")
    mu_imag = local_hadamard_test(alphas, l=l, lp=lp, j=j, part="Im")

    return mu_real + 1.0j * mu_imag

def cost(alphas):
    """Local version of the cost function. Tends to zero when A|x> is proportional to |b>."""

    # numerator
    mu_sum = 0.0

    for l in range(0, len(c_l)):
        for lp in range(0, len(c_l)):
            for j in range(0, n_qubits):
                mu_sum = mu_sum + c_l[l] * np.conj(c_l[lp]) * mu(alphas, l, lp, j)

    mu_sum = abs(mu_sum)

    # denominator
    norm = 0.0

    for l in range(0, len(c_l)):
        for lp in range(0, len(c_l)):
            norm = norm + c_l[l] * np.conj(c_l[lp]) * mu(alphas, l, lp, -1)

    norm = abs(norm)

    # Cost function C_L
    return 0.5 - 0.5 * mu_sum / (n_qubits * norm)


# Train the parametrized circuit
opt = qml.GradientDescentOptimizer(stepsize=step_size)
    
np.random.seed(rng_seed)
alphas = q_delta * np.random.randn(n_qubits, requires_grad=True)
print(f"Initial parameters: {alphas}")

for it in range(steps):
    alphas, cost_before_step = opt.step_and_cost(cost, alphas)
    log_metric(metric_name="cost_function", iteration_number=it, value=cost_before_step)
    log_metric(metric_name="circuit_invocations", iteration_number=it, value=circ_invoc)

print(f"Optimal alphas: {alphas}")

# Sample the bitstring probabilities from the variational circuit with optimal values of alphas
@qml.qnode(device_for_sampling, interface="autograd")
def prepare_and_sample(weights):
    global circ_invoc
    circ_invoc += 1

    # Variational circuit generating a guess for the solution vector |x>
    V(weights)

    # Probability of each computational basis state.
    return qml.probs()


q_probs = prepare_and_sample(alphas)
print("|<i|x>|^2 =", q_probs)

task_tracker.stop()

save_job_result({
    'alphas': alphas.tolist(),
    'q_probs': q_probs.tolist(),
    'braket_tasks': task_tracker.quantum_tasks_statistics(),
    'circuit_invocations': circ_invoc,
})
print("Execution completed")