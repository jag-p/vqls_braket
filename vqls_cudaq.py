import cudaq
import numpy as np
from typing import List
from braket.jobs.metrics import log_metric
from braket.jobs import save_job_result
from braket.jobs.environment_variables import get_hyperparameters


print("Quantum solution")
hyperparameters = get_hyperparameters()
print(hyperparameters)
n_qubits = int(hyperparameters["n_qubits"])
n_shots = int(hyperparameters["n_shots"])
steps = int(hyperparameters["steps"])
q_delta = float(hyperparameters["q_delta"])


@cudaq.kernel
def c_a(idx: int, control_qubit: cudaq.qubit, target_qubit_1: cudaq.qubit, target_qubit_2: cudaq.qubit):
    if idx == 1:
        x.ctrl(control_qubit, target_qubit_1)
        z.ctrl(control_qubit, target_qubit_2)
    elif idx == 2:
        x.ctrl(control_qubit, target_qubit_1)


@cudaq.kernel
def u_b(qvector: cudaq.qvector, qubit_count: int):
    for idx in range(qubit_count):
        h(qvector[idx])


@cudaq.kernel
def variational_block(weights: List[float], qvector: cudaq.qvector, qubit_count: int):
    # We first prepare an equal superposition of all the states of the computational basis.
    for idx in range(qubit_count):
        h(qvector[idx])
    # A very minimal variational circuit.
    for idx, element in enumerate(weights):
        ry(element, qvector[idx])


@cudaq.kernel
def local_hadamard_test(weights: List[float], l: int, lp: int, j: int, part: int, qubit_count: int):
    qvector = cudaq.qvector(qubit_count + 1)
    ancilla_qubit = qvector[qubit_count]

    # First Hadamard gate applied to the ancilla qubit
    h(ancilla_qubit)

    # For estimating the imaginary part of the coefficient "mu", we must add a "-i" phase gate.
    if part == 1:
        # u3(0, np.pi / 2, 0, ancilla_idx)
        s.adj(ancilla_qubit)

    # Variational circuit generating a guess for the solution vector |x>
    variational_block(weights, qvector, qubit_count)

    # Controlled application of the unitary component A_l of the problem matrix A.
    c_a(l, ancilla_qubit, qvector[0], qvector[1])

    # Adjoint of the unitary U_b associated to the problem vector |b>. In this specific example Adjoint(U_b) = U_b.
    u_b(qvector, qubit_count)

    # Controlled Z operator at position j. If j = -1, apply the identity.
    if j != -1:
        z.ctrl(ancilla_qubit, qvector[j])

    # Unitary U_b associated to the problem vector |b>.
    u_b(qvector, qubit_count)

    # Controlled application of Adjoint(A_lp). In this specific example Adjoint(A_lp) = A_lp.
    c_a(lp, ancilla_qubit, qvector[0], qvector[1])

    # Second Hadamard gate applied to the ancillary qubit.
    h(ancilla_qubit)


@cudaq.kernel
def prepare(weights: List[float], qubit_count: int):
    qvector = cudaq.qvector(qubit_count)
    variational_block(weights, qvector, qubit_count)


def mu(weights, l=None, lp=None, j=None):
    """Generates the coefficients to compute the "local" cost function C_L."""
    hamiltonian = cudaq.spin.z(n_qubits)

    mu_real = cudaq.observe(local_hadamard_test, hamiltonian, weights, l, lp, j, 0, n_qubits, shots_count=0).expectation()
    mu_imag = cudaq.observe(local_hadamard_test, hamiltonian, weights, l, lp, j, 1, n_qubits, shots_count=0).expectation()

    return mu_real + 1.0j * mu_imag


c = np.array([1.0, 0.2, 0.2])
iteration = 0


def psi_norm(weights):
    """Returns the normalization constant <psi|psi>, where |psi> = A |x>."""
    norm = 0.0

    for l in range(0, len(c)):
        for lp in range(0, len(c)):
            norm = norm + c[l] * np.conj(c[lp]) * mu(weights, l, lp, -1)

    return abs(norm)


def cost_loc(weights):
    """Local version of the cost function. Tends to zero when A|x> is proportional to |b>."""
    global iteration
    iteration += 1
    mu_sum = 0.0


    for l in range(0, len(c)):
        for lp in range(0, len(c)):
            for j in range(0, n_qubits):
                mu_sum = mu_sum + c[l] * np.conj(c[lp]) * mu(weights, l, lp, j)

    mu_sum = abs(mu_sum)

    # Cost function C_L
    cost = 0.5 - 0.5 * mu_sum / (n_qubits * psi_norm(weights))
    log_metric(metric_name="cost", iteration_number=iteration, value=cost)
    # log_metric(metric_name="cost", value=cost)

    return cost



# Specify the optimizer and its initial parameters.
cudaq.set_random_seed(42)
optimizer = cudaq.optimizers.COBYLA()
np.random.seed(42)
optimizer.initial_parameters = [0.0016525, 0.04126595, 0.00084824] #q_delta * np.random.randn(n_qubits)
print("Initial parameters = ", optimizer.initial_parameters)

# Optimize!
optimizer.max_iterations = steps
optimal_expectation, optimal_parameters = optimizer.optimize(dimensions=n_qubits, function=cost_loc)

print("Optimal value = ", optimal_expectation)
print("Optimal parameters = ", optimal_parameters)

raw_samples = cudaq.sample(prepare, optimal_parameters, n_qubits, shots_count=n_shots)
raw_samples = dict(sorted(raw_samples.items()))
print("Raw samples = ", raw_samples)

samples = {}
for sam, count in raw_samples.items():
    samples[int("".join(str(bs) for bs in sam), base=2)] = count

samples = dict(sorted(samples.items()))
print("Samples = ", samples)

q_probs = [count / n_shots for count in samples.values()]

print("|<x|n>|^2=\n", q_probs)

save_job_result({
    "raw_samples": raw_samples,
    "samples": samples,
    "q_probs": q_probs
})