from dataclasses import dataclass
import math

import numpy as np


PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def pauli_string_matrix(pauli):
    """Return a Pauli matrix with character i acting on Qiskit qubit i."""
    matrices = [PAULI[char] for char in reversed(str(pauli).upper())]
    if not matrices:
        raise ValueError("A Pauli string must contain at least one character.")
    result = matrices[0]
    for matrix in matrices[1:]:
        result = np.kron(result, matrix)
    return result


def tfim_terms(n_qubits, g, pbc=False):
    """Return H = -sum ZZ - g sum X in little-endian Pauli notation."""
    n_qubits = int(n_qubits)
    g = float(g)
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive.")
    if not math.isfinite(g):
        raise ValueError("g must be finite.")

    terms = []
    for qubit in range(n_qubits):
        chars = ["I"] * n_qubits
        chars[qubit] = "X"
        terms.append((-g, "".join(chars)))

        if qubit == n_qubits - 1:
            if not pbc or n_qubits < 2:
                continue
            neighbor = 0
        else:
            neighbor = qubit + 1
        chars = ["I"] * n_qubits
        chars[qubit] = "Z"
        chars[neighbor] = "Z"
        terms.append((-1.0, "".join(chars)))
    return terms


def tfim_hamiltonian(n_qubits, g, pbc=False):
    dim = 2 ** int(n_qubits)
    matrix = np.zeros((dim, dim), dtype=complex)
    for coefficient, pauli in tfim_terms(n_qubits, g, pbc):
        matrix += coefficient * pauli_string_matrix(pauli)
    return matrix


@dataclass(frozen=True)
class SpectralScaling:
    eta: float
    energy_low: float
    energy_high: float
    c1: float
    c2: float

    def rescale_energy(self, energy):
        return self.c1 * float(energy) + self.c2

    def to_dict(self):
        return {
            "method": "exact_spectrum",
            "eta": self.eta,
            "energy_low": self.energy_low,
            "energy_high": self.energy_high,
            "c1": self.c1,
            "c2": self.c2,
            "rescaled_bounds": [self.eta, math.pi - self.eta],
        }


def exact_spectral_scaling(hamiltonian, eta):
    eta = float(eta)
    if not 0.0 < eta < math.pi / 2.0:
        raise ValueError("eta must lie in (0, pi/2).")
    spectrum = np.linalg.eigvalsh(np.asarray(hamiltonian, dtype=complex))
    energy_low = float(np.min(spectrum))
    energy_high = float(np.max(spectrum))
    if energy_high - energy_low <= 1.0e-14:
        raise ValueError("The Hamiltonian has no nontrivial spectral width.")
    c1 = (math.pi - 2.0 * eta) / (energy_high - energy_low)
    c2 = eta - c1 * energy_low
    return SpectralScaling(
        eta=eta,
        energy_low=energy_low,
        energy_high=energy_high,
        c1=float(c1),
        c2=float(c2),
    )


def exact_ground_data(hamiltonian, atol=1.0e-10):
    values, vectors = np.linalg.eigh(np.asarray(hamiltonian, dtype=complex))
    order = np.argsort(np.real(values))
    values = np.real(values[order])
    vectors = vectors[:, order]
    tolerance = float(atol) * max(1.0, float(np.max(np.abs(values))))
    ground_mask = np.abs(values - values[0]) <= tolerance
    excited = values[~ground_mask]
    gap = None if excited.size == 0 else float(excited[0] - values[0])
    ground_space = vectors[:, ground_mask]
    projector = ground_space @ ground_space.conj().T
    return {
        "spectrum": values,
        "ground_energy": float(values[0]),
        "gap": gap,
        "ground_degeneracy": int(np.count_nonzero(ground_mask)),
        "ground_space": ground_space,
        "ground_projector": projector,
    }


def tfim_observables(n_qubits, pbc=False):
    """Return sum ZZ and sum X observables used by QETU-main's estimator."""
    n_qubits = int(n_qubits)
    dim = 2**n_qubits
    x_sum = np.zeros((dim, dim), dtype=complex)
    zz_sum = np.zeros((dim, dim), dtype=complex)
    for qubit in range(n_qubits):
        chars = ["I"] * n_qubits
        chars[qubit] = "X"
        x_sum += pauli_string_matrix("".join(chars))

        if qubit == n_qubits - 1:
            if not pbc or n_qubits < 2:
                continue
            neighbor = 0
        else:
            neighbor = qubit + 1
        chars = ["I"] * n_qubits
        chars[qubit] = "Z"
        chars[neighbor] = "Z"
        zz_sum += pauli_string_matrix("".join(chars))
    return zz_sum, x_sum
