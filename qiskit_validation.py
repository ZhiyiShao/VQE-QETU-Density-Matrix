from pathlib import Path
import importlib.util
import sys

import numpy as np


ONE_QUBIT_GATES = ("u1", "u2", "u3", "x", "rz", "rx", "h")
TWO_QUBIT_GATES = ("cx", "cy", "cz", "rzz")


def build_noise_model(depolarizing_rate, one_qubit_rate_factor=0.1):
    from qiskit.providers.aer.noise import NoiseModel, depolarizing_error

    rate = float(depolarizing_rate)
    factor = float(one_qubit_rate_factor)
    if not np.isfinite(rate) or not 0.0 <= rate <= 1.0:
        raise ValueError("depolarizing_rate must lie in [0, 1].")
    if not np.isfinite(factor) or not 0.0 < factor <= 1.0:
        raise ValueError("one_qubit_rate_factor must lie in (0, 1].")
    if rate == 0.0:
        return None
    model = NoiseModel()
    model.add_all_qubit_quantum_error(
        depolarizing_error(factor * rate, 1),
        list(ONE_QUBIT_GATES),
    )
    model.add_all_qubit_quantum_error(
        depolarizing_error(rate, 2),
        list(TWO_QUBIT_GATES),
    )
    return model


def build_native_circuit(model, trainable_phases):
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(model.n_total)
    for operation in model.circuit_operations(trainable_phases):
        append_gate(circuit, operation)
    return circuit


def append_gate(circuit, operation):
    name = operation.name
    qubits = list(operation.qubits)
    if name == "x":
        circuit.x(*qubits)
    elif name == "h":
        circuit.h(*qubits)
    elif name == "rx":
        circuit.rx(float(operation.angle), *qubits)
    elif name == "rz":
        circuit.rz(float(operation.angle), *qubits)
    elif name == "rzz":
        circuit.rzz(float(operation.angle), *qubits)
    elif name == "cy":
        circuit.cy(*qubits)
    elif name == "cz":
        circuit.cz(*qubits)
    else:
        raise ValueError(f"Unsupported Qiskit gate {name!r}.")


def build_qetu_main_circuit(model, trainable_phases, qetu_repo):
    qetu = load_qetu_module(qetu_repo)
    tfim = qetu.transverse_field_Ising_model(
        n_qubits=model.n_system,
        g_coupling=model.g,
        pbc=model.pbc,
    )
    trotter = tfim.build_trotter(
        t_tot=model.evolution_time,
        n_segments=model.trotter_steps,
        trotter_order=model.trotter_order,
    )
    block = qetu.HadamardBlockEncoding()
    block.build_from_unitary_TFIM(qc=trotter, shift=model.shift_angle)

    full = model.expand_trainable_phases_numpy(trainable_phases)
    circuit = tfim.apply_qsvt_xrot(
        be_qc=block.qc,
        phi_seq_su2=full,
        init_state=model.initial_bitstring,
        measure=False,
        measure_full=False,
    )
    return qetu, tfim, circuit


def density_matrix_aer(
    circuit,
    depolarizing_rate=0.0,
    one_qubit_rate_factor=0.1,
    seed_simulator=101,
):
    from qiskit import execute
    from qiskit.providers.aer import AerSimulator

    simulation = circuit.copy()
    simulation.save_density_matrix()
    backend = AerSimulator(method="density_matrix")
    result = execute(
        simulation,
        backend,
        shots=1,
        noise_model=build_noise_model(
            depolarizing_rate,
            one_qubit_rate_factor,
        ),
        optimization_level=0,
        seed_simulator=int(seed_simulator),
    ).result()
    return np.asarray(result.data(0)["density_matrix"], dtype=complex)


def validate_torch_implementation(
    model,
    trainable_phases,
    depolarizing_rate,
    qetu_repo,
    atol_noiseless=1.0e-10,
    atol_noisy=2.0e-9,
):
    phases = np.asarray(trainable_phases, dtype=float).reshape(-1)
    native = build_native_circuit(model, phases)
    _, _, upstream = build_qetu_main_circuit(model, phases, qetu_repo)

    native_ideal = density_matrix_aer(native, 0.0)
    upstream_ideal = density_matrix_aer(upstream, 0.0)
    upstream_error = float(np.max(np.abs(native_ideal - upstream_ideal)))

    torch_result = model.simulate(
        phases,
        depolarizing_rate=float(depolarizing_rate),
        measurement_basis_noise=True,
    )
    torch_density = torch_result["rho_full"].detach().cpu().numpy()
    aer_noisy = density_matrix_aer(
        native,
        depolarizing_rate=float(depolarizing_rate),
        one_qubit_rate_factor=model.one_qubit_rate_factor,
    )
    noisy_error = float(np.max(np.abs(torch_density - aer_noisy)))
    upstream_noisy = density_matrix_aer(
        upstream,
        depolarizing_rate=float(depolarizing_rate),
        one_qubit_rate_factor=model.one_qubit_rate_factor,
    )
    upstream_noisy_error = float(
        np.max(np.abs(torch_density - upstream_noisy))
    )

    hadamard = native.copy()
    for qubit in range(1, model.n_total):
        hadamard.h(qubit)
    aer_hadamard = density_matrix_aer(
        hadamard,
        depolarizing_rate=float(depolarizing_rate),
        one_qubit_rate_factor=model.one_qubit_rate_factor,
    )
    comp_system, _ = postselect_ancilla_zero(aer_noisy)
    had_system, _ = postselect_ancilla_zero(aer_hadamard)
    zz_sum = model.zz_sum.detach().cpu().numpy()
    z_sum = model.z_sum.detach().cpu().numpy()
    aer_measurement_energy = float(
        -np.real(np.trace(zz_sum @ comp_system))
        - model.g * np.real(np.trace(z_sum @ had_system))
    )
    torch_measurement_energy = float(
        torch_result["energy_measurement_model"].detach().cpu()
    )
    measurement_error = abs(aer_measurement_energy - torch_measurement_energy)

    report = {
        "qetu_main_noiseless_max_abs_error": upstream_error,
        "aer_noisy_max_abs_error": noisy_error,
        "qetu_main_aer_noisy_max_abs_error": upstream_noisy_error,
        "aer_measurement_energy": aer_measurement_energy,
        "torch_measurement_energy": torch_measurement_energy,
        "measurement_energy_abs_error": measurement_error,
        "noiseless_tolerance": float(atol_noiseless),
        "noisy_tolerance": float(atol_noisy),
        "qetu_main_noiseless_passed": bool(upstream_error <= atol_noiseless),
        "aer_noisy_passed": bool(noisy_error <= atol_noisy),
        "qetu_main_aer_noisy_passed": bool(
            upstream_noisy_error <= atol_noisy
        ),
        "measurement_energy_passed": bool(measurement_error <= atol_noisy),
        "native_circuit": circuit_summary(native),
        "qetu_main_circuit": circuit_summary(upstream),
    }
    report["passed"] = bool(
        report["qetu_main_noiseless_passed"]
        and report["aer_noisy_passed"]
        and report["qetu_main_aer_noisy_passed"]
        and report["measurement_energy_passed"]
    )
    return report


def finite_shot_evaluation(
    model,
    trainable_phases,
    qetu_repo,
    depolarizing_rate,
    shots_per_basis,
    repetitions,
    seed_simulator=200001,
):
    from qiskit import execute
    from qiskit.providers.aer import AerSimulator

    qetu, _, circuit = build_qetu_main_circuit(
        model,
        trainable_phases,
        qetu_repo,
    )
    comp, had = qetu.circuit_energy_VQEType(circuit)
    backend = AerSimulator()
    noise_model = build_noise_model(
        depolarizing_rate,
        model.one_qubit_rate_factor,
    )
    energies = []
    probabilities = []
    invalid_repetitions = []
    for repetition in range(int(repetitions)):
        seed = int(seed_simulator) + 2 * repetition
        comp_counts = execute(
            comp,
            backend,
            shots=int(shots_per_basis),
            noise_model=noise_model,
            optimization_level=0,
            seed_simulator=seed,
        ).result().get_counts(comp)
        had_counts = execute(
            had,
            backend,
            shots=int(shots_per_basis),
            noise_model=noise_model,
            optimization_level=0,
            seed_simulator=seed + 1,
        ).result().get_counts(had)
        comp_probability = ancilla_zero_probability(comp_counts)
        had_probability = ancilla_zero_probability(had_counts)
        probabilities.append(0.5 * (comp_probability + had_probability))
        if comp_probability == 0.0 or had_probability == 0.0:
            invalid_repetitions.append(
                {
                    "repetition": int(repetition),
                    "computational_basis_success_probability": float(
                        comp_probability
                    ),
                    "hadamard_basis_success_probability": float(
                        had_probability
                    ),
                }
            )
            continue
        energy_zz = qetu.counts_to_energy(comp_counts, "zz")
        energy_x = qetu.counts_to_energy(had_counts, "x")
        energies.append(float(-energy_zz - model.g * energy_x))

    energies = np.asarray(energies, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if energies.size == 0:
        raise ValueError(
            "No repetition contained postselected samples in both "
            "measurement bases. Increase shots_per_basis or optimize for a "
            "larger ancilla-0 success probability."
        )
    exact_energy = float(model.ground_data["ground_energy"])
    return {
        "method": "QETU-main.measure_energy_VQEType",
        "backend": "AerSimulator",
        "depolarizing_rate": float(depolarizing_rate),
        "shots_per_basis": int(shots_per_basis),
        "repetitions": int(repetitions),
        "valid_energy_repetitions": int(energies.size),
        "invalid_energy_repetitions": int(len(invalid_repetitions)),
        "invalid_repetition_details": invalid_repetitions,
        "total_shots": int(2 * shots_per_basis * repetitions),
        "energy_samples": energies.tolist(),
        "energy_mean": float(np.mean(energies)),
        "energy_standard_deviation": _sample_std(energies),
        "absolute_error_of_mean": float(abs(np.mean(energies) - exact_energy)),
        "success_probability_samples": probabilities.tolist(),
        "success_probability_mean": float(np.mean(probabilities)),
        "success_probability_standard_deviation": _sample_std(probabilities),
        "circuit": circuit_summary(circuit),
    }


def load_qetu_module(qetu_repo):
    path = Path(qetu_repo).resolve() / "qet_u.py"
    if not path.exists():
        raise FileNotFoundError(path)
    name = "_vqe_qsp_density_qetu"
    existing = sys.modules.get(name)
    if existing is not None and Path(existing.__file__).resolve() == path:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import QETU from {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def ancilla_zero_probability(counts):
    total = sum(int(count) for count in counts.values())
    successful = sum(
        int(count)
        for bitstring, count in counts.items()
        if str(bitstring).replace(" ", "")[-1] == "0"
    )
    return float(successful / total)


def postselect_ancilla_zero(full_density):
    full_density = np.asarray(full_density, dtype=complex)
    system = full_density[0::2, 0::2]
    probability = float(np.real(np.trace(system)))
    if probability <= 1.0e-15:
        raise ValueError("Postselection probability for ancilla=0 is zero.")
    return system / probability, probability


def circuit_summary(circuit):
    return {
        "num_qubits": int(circuit.num_qubits),
        "depth": int(circuit.depth()),
        "size": int(circuit.size()),
        "count_ops": {
            str(name): int(count)
            for name, count in dict(circuit.count_ops()).items()
        },
    }


def _sample_std(values):
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
