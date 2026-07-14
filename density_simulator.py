from dataclasses import dataclass
from itertools import product
import math

import numpy as np
import torch


@dataclass(frozen=True)
class GateSpec:
    name: str
    qubits: tuple
    angle: float = None

    @property
    def arity(self):
        return len(self.qubits)

    def inverse(self):
        if self.name in ("rx", "rz", "rzz"):
            return GateSpec(self.name, self.qubits, -float(self.angle))
        if self.name in ("x", "h", "cy", "cz"):
            return self
        raise ValueError(f"No inverse rule for gate {self.name!r}.")


I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2.0)


def rx_matrix(angle):
    half = 0.5 * angle
    return math.cos(half) * I2 - 1j * math.sin(half) * X


def rz_matrix(angle):
    half = 0.5 * angle
    return np.diag([np.exp(-1j * half), np.exp(1j * half)])


def rzz_matrix(angle):
    half = 0.5 * angle
    return np.diag(
        [
            np.exp(-1j * half),
            np.exp(1j * half),
            np.exp(1j * half),
            np.exp(-1j * half),
        ]
    )


def controlled_matrix(target):
    """Return controlled-target with local bit 0 as control and bit 1 as target."""
    result = np.zeros((4, 4), dtype=complex)
    for column in range(4):
        control = column & 1
        target_in = (column >> 1) & 1
        if control == 0:
            result[column, column] = 1.0
            continue
        for target_out in range(2):
            row = 1 | (target_out << 1)
            result[row, column] = target[target_out, target_in]
    return result


def local_gate_matrix(spec):
    name = spec.name
    if name == "x":
        return X
    if name == "h":
        return H
    if name == "rx":
        return rx_matrix(float(spec.angle))
    if name == "rz":
        return rz_matrix(float(spec.angle))
    if name == "rzz":
        return rzz_matrix(float(spec.angle))
    if name == "cy":
        return controlled_matrix(Y)
    if name == "cz":
        return controlled_matrix(Z)
    raise ValueError(f"Unsupported gate {name!r}.")


def embed_operator(local_operator, qubits, n_qubits):
    """Embed a local operator using Qiskit's little-endian qargs convention."""
    qubits = tuple(int(qubit) for qubit in qubits)
    n_qubits = int(n_qubits)
    if len(set(qubits)) != len(qubits):
        raise ValueError("Gate qubits must be distinct.")
    if any(qubit < 0 or qubit >= n_qubits for qubit in qubits):
        raise ValueError("Gate qubit lies outside the register.")
    local = np.asarray(local_operator, dtype=complex)
    local_dim = 2 ** len(qubits)
    if local.shape != (local_dim, local_dim):
        raise ValueError("Local operator has the wrong dimension.")

    dim = 2**n_qubits
    full = np.zeros((dim, dim), dtype=complex)
    for column in range(dim):
        local_column = sum(
            ((column >> qubit) & 1) << position
            for position, qubit in enumerate(qubits)
        )
        for local_row in range(local_dim):
            amplitude = local[local_row, local_column]
            if abs(amplitude) == 0.0:
                continue
            row = column
            for position, qubit in enumerate(qubits):
                bit = (local_row >> position) & 1
                row = (row & ~(1 << qubit)) | (bit << qubit)
            full[row, column] += amplitude
    return full


def gate_matrix(spec, n_qubits):
    return embed_operator(local_gate_matrix(spec), spec.qubits, n_qubits)


def torch_rx(angle, dtype=torch.complex128, device=None):
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    angle = torch.as_tensor(angle, dtype=real_dtype, device=device)
    half = 0.5 * angle
    cosine = torch.cos(half).to(dtype)
    sine = torch.sin(half).to(dtype)
    return torch.stack(
        (
            torch.stack((cosine, -1j * sine)),
            torch.stack((-1j * sine, cosine)),
        )
    )


def unitary_superoperator(unitary):
    """Return the row-vectorized map rho -> U rho U^dagger."""
    return torch.kron(unitary, unitary.conj())


def apply_superoperator(rho, superoperator):
    dim = rho.shape[0]
    return (superoperator @ rho.reshape(-1)).reshape(dim, dim)


class GatewiseDepolarizingSimulator:
    """Small-system density simulator matching Qiskit's depolarizing channel."""

    def __init__(
        self,
        n_qubits,
        depolarizing_rate,
        one_qubit_rate_factor=0.1,
        dtype=torch.complex128,
        device="cpu",
    ):
        self.n_qubits = int(n_qubits)
        self.depolarizing_rate = _rate(depolarizing_rate)
        self.one_qubit_rate_factor = float(one_qubit_rate_factor)
        if not math.isfinite(self.one_qubit_rate_factor) or not (
            0.0 < self.one_qubit_rate_factor <= 1.0
        ):
            raise ValueError("one_qubit_rate_factor must lie in (0, 1].")
        self.dtype = dtype
        self.device = torch.device(device)
        self.dim = 2**self.n_qubits
        self._depolarizing_cache = {}
        self._system_identity = torch.eye(
            2 ** (self.n_qubits - 1),
            dtype=self.dtype,
            device=self.device,
        )

    def gate_rate(self, arity):
        if int(arity) == 1:
            return self.one_qubit_rate_factor * self.depolarizing_rate
        if int(arity) == 2:
            return self.depolarizing_rate
        raise ValueError("Only one- and two-qubit noisy gates are supported.")

    def depolarizing_superoperator(self, qubits):
        qubits = tuple(int(qubit) for qubit in qubits)
        key = tuple(qubits)
        cached = self._depolarizing_cache.get(key)
        if cached is not None:
            return cached
        rate = self.gate_rate(len(qubits))
        identity = torch.eye(
            self.dim * self.dim,
            dtype=self.dtype,
            device=self.device,
        )
        if rate == 0.0:
            self._depolarizing_cache[key] = identity
            return identity

        paulis = (I2, X, Y, Z)
        twirl = torch.zeros_like(identity)
        for labels in product(range(4), repeat=len(qubits)):
            local = paulis[labels[0]]
            for label in labels[1:]:
                local = np.kron(paulis[label], local)
            full = embed_operator(local, qubits, self.n_qubits)
            full = torch.as_tensor(full, dtype=self.dtype, device=self.device)
            twirl = twirl + unitary_superoperator(full)
        twirl = twirl / float(4 ** len(qubits))
        channel = (1.0 - rate) * identity + rate * twirl
        self._depolarizing_cache[key] = channel
        return channel

    def operation_superoperator(self, spec):
        unitary = torch.as_tensor(
            gate_matrix(spec, self.n_qubits),
            dtype=self.dtype,
            device=self.device,
        )
        return self.depolarizing_superoperator(spec.qubits) @ unitary_superoperator(unitary)

    def compile(self, operations):
        total = torch.eye(
            self.dim * self.dim,
            dtype=self.dtype,
            device=self.device,
        )
        for operation in operations:
            total = self.operation_superoperator(operation) @ total
        return total

    def apply_dynamic_rx(self, rho, angle, qubit=0):
        if int(qubit) != 0:
            raise ValueError("The differentiable QETU phase rotation must act on ancilla 0.")
        unitary = torch.kron(
            self._system_identity,
            torch_rx(angle, self.dtype, self.device),
        )
        rho = unitary @ rho @ unitary.conj().T
        return apply_superoperator(rho, self.depolarizing_superoperator((0,)))


def pure_zero_density(n_qubits, dtype=torch.complex128, device="cpu"):
    dim = 2 ** int(n_qubits)
    rho = torch.zeros((dim, dim), dtype=dtype, device=device)
    rho[0, 0] = 1.0
    return rho


def density_diagnostics(rho):
    hermiticity = torch.max(torch.abs(rho - rho.conj().T))
    trace = torch.trace(rho)
    return {
        "trace_real": float(torch.real(trace).detach().cpu()),
        "trace_imag": float(torch.imag(trace).detach().cpu()),
        "hermiticity_error": float(hermiticity.detach().cpu()),
    }


def _rate(value):
    if isinstance(value, bool):
        raise ValueError("depolarizing_rate must be a real number.")
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("depolarizing_rate must lie in [0, 1].")
    return value
