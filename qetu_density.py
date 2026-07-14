import math

import numpy as np
import torch

from density_simulator import (
    GateSpec,
    GatewiseDepolarizingSimulator,
    apply_superoperator,
    pure_zero_density,
)
from tfim import (
    exact_ground_data,
    exact_spectral_scaling,
    tfim_hamiltonian,
    tfim_observables,
)


def wrap_phases(phases):
    return torch.remainder(phases + math.pi, 2.0 * math.pi) - math.pi


def expand_palindromic_phases(half_phases):
    half = torch.as_tensor(half_phases)
    if half.ndim != 1 or half.numel() < 2:
        raise ValueError("half_phases must be a one-dimensional half sequence.")
    return torch.cat((half, torch.flip(half[:-1], dims=(0,))))


class DifferentiableTFIMQETU:
    """Differentiable density-matrix mirror of QETU-main's TFIM circuit."""

    def __init__(
        self,
        n_qubits,
        g,
        pbc,
        eta,
        tau,
        trotter_steps,
        trotter_order,
        degree,
        phase_symmetry,
        initial_bitstring,
        one_qubit_rate_factor=0.1,
        dtype=torch.complex128,
        device="cpu",
    ):
        self.n_system = int(n_qubits)
        self.n_total = self.n_system + 1
        self.g = float(g)
        self.pbc = bool(pbc)
        self.tau = float(tau)
        self.trotter_steps = int(trotter_steps)
        self.trotter_order = int(trotter_order)
        self.degree = int(degree)
        self.phase_symmetry = str(phase_symmetry)
        self.initial_bitstring = str(initial_bitstring)
        self.one_qubit_rate_factor = float(one_qubit_rate_factor)
        self.dtype = dtype
        self.device = torch.device(device)

        self._validate()
        self.hamiltonian_numpy = tfim_hamiltonian(
            self.n_system,
            self.g,
            self.pbc,
        )
        self.ground_data = exact_ground_data(self.hamiltonian_numpy)
        self.scaling = exact_spectral_scaling(self.hamiltonian_numpy, eta)
        self.hamiltonian = torch.as_tensor(
            self.hamiltonian_numpy,
            dtype=self.dtype,
            device=self.device,
        )
        self.ground_projector = torch.as_tensor(
            self.ground_data["ground_projector"],
            dtype=self.dtype,
            device=self.device,
        )
        zz_sum, x_sum = tfim_observables(self.n_system, self.pbc)
        self.zz_sum = torch.as_tensor(zz_sum, dtype=self.dtype, device=self.device)
        self.x_sum = torch.as_tensor(x_sum, dtype=self.dtype, device=self.device)
        self.z_sum = self._build_z_sum()

        self.evolution_time = self.tau * self.scaling.c1
        self.shift_angle = self.tau * self.scaling.c2
        self.preparation_operations = self._build_preparation_operations()
        self.trotter_operations = self._build_trotter_operations()
        self.block_operations = self._build_block_operations()
        self._simulator_cache = {}
        self._compiled_cache = {}

    @property
    def num_trainable_phases(self):
        if self.phase_symmetry == "palindrome":
            return self.degree // 2 + 1
        return self.degree + 1

    def _validate(self):
        if self.n_system <= 0:
            raise ValueError("n_qubits must be positive.")
        if abs(self.tau - 0.5) > 1.0e-12:
            raise ValueError("QETU-main's TFIM benchmark requires tau=0.5.")
        if self.trotter_steps <= 0 or self.trotter_order not in (1, 2):
            raise ValueError("Invalid Trotter configuration.")
        if self.degree < 2 or self.degree % 2:
            raise ValueError("degree must be a positive even integer.")
        if self.phase_symmetry not in ("palindrome", "full"):
            raise ValueError("phase_symmetry must be palindrome or full.")
        if len(self.initial_bitstring) != self.n_system or set(self.initial_bitstring) - set("01"):
            raise ValueError("initial_bitstring must contain one bit per system qubit.")

    def _build_preparation_operations(self):
        return [
            GateSpec("x", (index + 1,))
            for index, bit in enumerate(self.initial_bitstring)
            if bit == "1"
        ]

    def _build_trotter_operations(self):
        dt = self.evolution_time / self.trotter_steps
        operations = []

        def append_x(time):
            for qubit in range(self.n_system):
                operations.append(
                    GateSpec("rx", (qubit + 1,), -2.0 * time * self.g)
                )

        def append_zz(time):
            for qubit in range(0, self.n_system - 1, 2):
                operations.append(
                    GateSpec("rzz", (qubit + 1, qubit + 2), -2.0 * time)
                )
            for qubit in range(1, self.n_system - 1, 2):
                operations.append(
                    GateSpec("rzz", (qubit + 1, qubit + 2), -2.0 * time)
                )
            if self.pbc:
                operations.append(
                    GateSpec("rzz", (self.n_system, 1), -2.0 * time)
                )

        append_x(dt if self.trotter_order == 1 else 0.5 * dt)
        for _ in range(self.trotter_steps - 1):
            append_zz(dt)
            append_x(dt)
        append_zz(dt)
        if self.trotter_order == 2:
            append_x(0.5 * dt)
        return operations

    def _build_block_operations(self):
        operations = [GateSpec("rz", (0,), -2.0 * self.shift_angle)]
        for system_index in range(self.n_system):
            gate = "cy" if system_index % 2 == 0 else "cz"
            operations.append(GateSpec(gate, (0, system_index + 1)))
        operations.extend(operation.inverse() for operation in reversed(self.trotter_operations))
        for system_index in range(self.n_system):
            gate = "cy" if system_index % 2 == 0 else "cz"
            operations.append(GateSpec(gate, (0, system_index + 1)))
        return operations

    def simulator(self, depolarizing_rate):
        rate = float(depolarizing_rate)
        key = rate
        if key not in self._simulator_cache:
            self._simulator_cache[key] = GatewiseDepolarizingSimulator(
                self.n_total,
                rate,
                one_qubit_rate_factor=self.one_qubit_rate_factor,
                dtype=self.dtype,
                device=self.device,
            )
        return self._simulator_cache[key]

    def _compiled_channels(self, depolarizing_rate):
        rate = float(depolarizing_rate)
        cached = self._compiled_cache.get(rate)
        if cached is not None:
            return cached
        simulator = self.simulator(rate)
        preparation = simulator.compile(self.preparation_operations)
        block = simulator.compile(self.block_operations)

        measurement_simulator = GatewiseDepolarizingSimulator(
            self.n_system,
            rate,
            one_qubit_rate_factor=self.one_qubit_rate_factor,
            dtype=self.dtype,
            device=self.device,
        )
        hadamards = [GateSpec("h", (qubit,)) for qubit in range(self.n_system)]
        measurement_h = measurement_simulator.compile(hadamards)
        cached = {
            "preparation": preparation,
            "block": block,
            "measurement_h": measurement_h,
        }
        self._compiled_cache[rate] = cached
        return cached

    def expand_trainable_phases(self, trainable_phases):
        trainable = torch.as_tensor(
            trainable_phases,
            dtype=torch.float64,
            device=self.device,
        )
        if trainable.ndim != 1 or trainable.numel() != self.num_trainable_phases:
            raise ValueError(
                f"Expected {self.num_trainable_phases} trainable phases."
            )
        wrapped = wrap_phases(trainable)
        if self.phase_symmetry == "palindrome":
            return expand_palindromic_phases(wrapped)
        return wrapped

    def expand_trainable_phases_numpy(self, trainable_phases):
        trainable = np.asarray(trainable_phases, dtype=float).reshape(-1)
        if trainable.size != self.num_trainable_phases:
            raise ValueError(
                f"Expected {self.num_trainable_phases} trainable phases."
            )
        wrapped = (trainable + math.pi) % (2.0 * math.pi) - math.pi
        if self.phase_symmetry == "palindrome":
            return np.concatenate((wrapped, wrapped[-2::-1]))
        return wrapped

    def simulate(
        self,
        trainable_phases,
        depolarizing_rate,
        measurement_basis_noise=True,
    ):
        phases = self.expand_trainable_phases(trainable_phases)
        offsets = torch.zeros_like(phases)
        offsets[0] = math.pi / 4.0
        offsets[-1] = -math.pi / 4.0
        effective = phases + offsets

        simulator = self.simulator(depolarizing_rate)
        channels = self._compiled_channels(depolarizing_rate)
        rho = pure_zero_density(self.n_total, self.dtype, self.device)
        rho = apply_superoperator(rho, channels["preparation"])
        rho = simulator.apply_dynamic_rx(rho, -2.0 * effective[-1], qubit=0)
        for phase_index in range(self.degree - 1, -1, -1):
            rho = apply_superoperator(rho, channels["block"])
            rho = simulator.apply_dynamic_rx(
                rho,
                -2.0 * effective[phase_index],
                qubit=0,
            )

        system_unnormalized = rho[0::2, 0::2]
        success_probability = torch.real(torch.trace(system_unnormalized))
        denominator = torch.clamp(success_probability, min=1.0e-12)
        system_density = system_unnormalized / denominator
        state_energy = _real_trace(self.hamiltonian @ system_density)
        fidelity = _real_trace(self.ground_projector @ system_density)

        zz_expectation = _real_trace(self.zz_sum @ system_density)
        if measurement_basis_noise:
            measured_x_density = apply_superoperator(
                system_density,
                channels["measurement_h"],
            )
            x_expectation = _real_trace(self.z_sum @ measured_x_density)
        else:
            x_expectation = _real_trace(self.x_sum @ system_density)
        measurement_energy = -zz_expectation - self.g * x_expectation

        return {
            "rho_full": rho,
            "rho_system": system_density,
            "phi_su2": phases,
            "success_probability_ancilla0": success_probability,
            "energy_state": state_energy,
            "energy_measurement_model": measurement_energy,
            "fidelity_ground_space": fidelity,
            "zz_expectation": zz_expectation,
            "x_expectation": x_expectation,
        }

    def _build_z_sum(self):
        dim = 2**self.n_system
        result = np.zeros((dim, dim), dtype=complex)
        from tfim import pauli_string_matrix

        for qubit in range(self.n_system):
            chars = ["I"] * self.n_system
            chars[qubit] = "Z"
            result += pauli_string_matrix("".join(chars))
        return torch.as_tensor(result, dtype=self.dtype, device=self.device)

    def circuit_operations(self, trainable_phases):
        phases = self.expand_trainable_phases_numpy(trainable_phases)
        effective = phases.copy()
        effective[0] += math.pi / 4.0
        effective[-1] -= math.pi / 4.0
        operations = list(self.preparation_operations)
        operations.append(GateSpec("rx", (0,), -2.0 * effective[-1]))
        for phase_index in range(self.degree - 1, -1, -1):
            operations.extend(self.block_operations)
            operations.append(
                GateSpec("rx", (0,), -2.0 * effective[phase_index])
            )
        return operations

    def summary(self):
        return {
            "model": {
                "name": "tfim",
                "n_qubits": self.n_system,
                "g": self.g,
                "pbc": self.pbc,
                "exact_spectrum": self.ground_data["spectrum"].tolist(),
                "exact_ground_energy": self.ground_data["ground_energy"],
                "exact_gap": self.ground_data["gap"],
                "ground_degeneracy": self.ground_data["ground_degeneracy"],
            },
            "spectral_scaling": self.scaling.to_dict(),
            "oracle": {
                "implementation": "qetu_main_tfim_control_free",
                "tau": self.tau,
                "trotter_order": self.trotter_order,
                "trotter_steps": self.trotter_steps,
                "evolution_time": self.evolution_time,
                "shift_angle": self.shift_angle,
                "trotter_gate_count": len(self.trotter_operations),
                "block_gate_count": len(self.block_operations),
            },
            "ansatz": {
                "degree": self.degree,
                "num_phases": self.degree + 1,
                "num_trainable_phases": self.num_trainable_phases,
                "symmetry": self.phase_symmetry,
                "phase_convention": "qetu_su2",
            },
        }


def metrics_to_python(metrics):
    keys = (
        "success_probability_ancilla0",
        "energy_state",
        "energy_measurement_model",
        "fidelity_ground_space",
        "zz_expectation",
        "x_expectation",
    )
    return {
        key: float(torch.real(metrics[key]).detach().cpu())
        for key in keys
    }


def _real_trace(matrix):
    return torch.real(torch.trace(matrix))
