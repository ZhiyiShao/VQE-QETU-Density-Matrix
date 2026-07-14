from pathlib import Path
import math

import numpy as np
import torch

from density_simulator import GateSpec, gate_matrix
from io_utils import load_yaml
from qetu_density import DifferentiableTFIMQETU, expand_palindromic_phases
from qiskit_validation import append_gate, validate_torch_implementation
from train_degree_scan import make_run_config, validate_scan_config
from trainer import DirectDensityMatrixTrainer, initialize_phases


ROOT = Path(__file__).resolve().parent
QETU_REPO = ROOT.parent / "QETU-main"


def main():
    print("Running VQE-QSP density-matrix checks")
    check_spectrum_and_scaling()
    check_phase_symmetry()
    check_full_phase_parameterization()
    check_gate_embedding()
    model = build_model()
    check_noiseless_energy_identity(model)
    check_autograd_against_finite_difference(model)
    check_degree_scan_protocol(model)
    check_report_snapshots()
    check_qetu_and_aer(model)
    check_short_training(model)
    print("All density-matrix checks passed.")


def build_model(phase_symmetry="palindrome"):
    return DifferentiableTFIMQETU(
        n_qubits=2,
        g=4.0,
        pbc=False,
        eta=0.1,
        tau=0.5,
        trotter_steps=3,
        trotter_order=2,
        degree=12,
        phase_symmetry=phase_symmetry,
        initial_bitstring="00",
        one_qubit_rate_factor=0.1,
    )


def check_spectrum_and_scaling():
    model = build_model()
    expected = -math.sqrt(65.0)
    assert abs(model.ground_data["ground_energy"] - expected) < 1.0e-12
    assert abs(model.scaling.rescale_energy(expected) - 0.1) < 1.0e-12
    assert abs(
        model.scaling.rescale_energy(model.scaling.energy_high)
        - (math.pi - 0.1)
    ) < 1.0e-12
    print("  PASS TFIM spectrum and exact spectral scaling")


def check_phase_symmetry():
    independent = torch.arange(7, dtype=torch.float64)
    full = expand_palindromic_phases(independent)
    assert full.numel() == 13
    assert torch.equal(full, torch.flip(full, dims=(0,)))
    print("  PASS palindromic SU(2) phase expansion")


def check_full_phase_parameterization():
    half = torch.tensor(
        np.random.default_rng(2).uniform(-0.25, 0.25, 7),
        dtype=torch.float64,
    )
    full = expand_palindromic_phases(half)
    symmetric_model = build_model("palindrome")
    full_model = build_model("full")
    symmetric = symmetric_model.simulate(half, 1.0e-3)["rho_full"]
    unconstrained = full_model.simulate(full, 1.0e-3)["rho_full"]
    assert torch.max(torch.abs(symmetric - unconstrained)) < 1.0e-12
    assert full_model.num_trainable_phases == 13
    endpoints, _ = initialize_phases(
        full_model,
        {"method": "pi_quarter_endpoints"},
        seed=0,
    )
    assert endpoints[0] == math.pi / 4.0
    assert endpoints[-1] == math.pi / 4.0
    print("  PASS full and palindromic parameterizations agree at the same phases")


def check_gate_embedding():
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator

    operations = (
        GateSpec("rx", (0,), 0.37),
        GateSpec("rz", (2,), -0.22),
        GateSpec("rzz", (1, 2), 0.41),
        GateSpec("cy", (0, 1)),
        GateSpec("cz", (0, 2)),
    )
    for operation in operations:
        circuit = QuantumCircuit(3)
        append_gate(circuit, operation)
        expected = Operator(circuit).data
        actual = gate_matrix(operation, 3)
        assert np.max(np.abs(actual - expected)) < 1.0e-12
    print("  PASS little-endian gate embedding against Qiskit")


def check_noiseless_energy_identity(model):
    phases = np.random.default_rng(3).uniform(-0.2, 0.2, 7)
    result = model.simulate(phases, 0.0, measurement_basis_noise=True)
    difference = abs(
        float(result["energy_state"].detach())
        - float(result["energy_measurement_model"].detach())
    )
    assert difference < 1.0e-11
    trace = torch.trace(result["rho_full"])
    assert abs(float(torch.real(trace)) - 1.0) < 1.0e-11
    print("  PASS noiseless state energy equals analytic measurement energy")


def check_autograd_against_finite_difference(model):
    phases = torch.tensor(
        np.random.default_rng(5).uniform(-0.25, 0.25, 7),
        dtype=torch.float64,
        requires_grad=True,
    )
    energy = model.simulate(phases, 1.0e-3)["energy_measurement_model"]
    energy.backward()
    analytic = float(phases.grad[2])
    epsilon = 1.0e-6
    plus = phases.detach().clone()
    minus = phases.detach().clone()
    plus[2] += epsilon
    minus[2] -= epsilon
    finite = float(
        (
            model.simulate(plus, 1.0e-3)["energy_measurement_model"]
            - model.simulate(minus, 1.0e-3)["energy_measurement_model"]
        )
        / (2.0 * epsilon)
    )
    assert abs(analytic - finite) <= 2.0e-6 * max(1.0, abs(finite))
    print(
        "  PASS PyTorch gradient against central finite difference "
        f"(autograd={analytic:+.6e}, finite={finite:+.6e})"
    )


def check_qetu_and_aer(model):
    phases = np.random.default_rng(7).uniform(-0.25, 0.25, 7)
    report = validate_torch_implementation(
        model,
        phases,
        1.0e-3,
        QETU_REPO,
    )
    assert report["passed"], report
    print(
        "  PASS QETU-main and Aer cross-validation "
        f"(ideal={report['qetu_main_noiseless_max_abs_error']:.2e}, "
        f"noisy={report['aer_noisy_max_abs_error']:.2e}, "
        f"measurement={report['measurement_energy_abs_error']:.2e})"
    )
    full_model = build_model("full")
    full_phases = np.random.default_rng(8).uniform(-0.25, 0.25, 13)
    full_report = validate_torch_implementation(
        full_model,
        full_phases,
        1.0e-3,
        QETU_REPO,
    )
    assert full_report["passed"], full_report
    print("  PASS full-phase QETU-main and Aer cross-validation")


def check_degree_scan_protocol(model):
    config_path = ROOT / "config" / "dm_tfim_n2_degree_scan_optimized.yml"
    config = load_yaml(config_path)
    validate_scan_config(config)
    ideal_run = make_run_config(
        config,
        ROOT / "results" / "check_optimized",
        12,
        0.0,
        19,
    )
    noisy_run = make_run_config(
        config,
        ROOT / "results" / "check_optimized",
        12,
        1.0e-3,
        19,
    )
    assert ideal_run["ansatz"]["symmetry"] == "full"
    assert ideal_run["initialization"]["method"] == "random_uniform"
    assert ideal_run["loss"]["success_penalty_weight"] == 0.0
    assert ideal_run["training"]["seed"] == noisy_run["training"]["seed"]
    assert ideal_run["noise"]["training_rate"] != noisy_run["noise"]["training_rate"]
    assert "pre_qsp_results" in ideal_run["paths"]
    assert ideal_run["training"]["polish"]["enabled"]

    phases = torch.tensor(
        np.random.default_rng(19).uniform(-0.25, 0.25, 7),
        dtype=torch.float64,
    )
    trainer = DirectDensityMatrixTrainer(
        model,
        ideal_run["training"],
        ideal_run["loss"],
        0.0,
    )
    _, _, details = trainer.objective(phases)
    assert float(details["success_penalty"].detach()) == 0.0
    print("  PASS optimized degree-scan protocol")


def check_report_snapshots():
    expected_degrees = list(range(2, 21, 2))
    expected_rates = [0.0, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2]
    for n_qubits in (2, 4):
        training = load_yaml(
            ROOT / "report_results" / f"tfim_n{n_qubits}_density_training.yml"
        )
        comparison = load_yaml(
            ROOT
            / "report_results"
            / f"tfim_n{n_qubits}_pre_qsp_comparison_shot20000.yml"
        )
        assert training["status"] == comparison["status"] == "complete"
        assert training["degrees"] == comparison["degrees"] == expected_degrees
        assert training["noise_rates"] == comparison["noise_rates"] == expected_rates
        assert len(training["records"]) == 50
        assert len(comparison["records"]) == 100
        assert not _absolute_strings(training)
        assert not _absolute_strings(comparison)
        assert not _contains_key(training, "independent_phases")
        assert not _contains_key(training, "num_independent_phases")
        assert not _contains_key(comparison, "independent_phases")
        assert set(comparison["methods"]) == {
            "pre_qsp",
            "density_matrix_training",
        }
        assert int(training["config"]["model"]["n_qubits"]) == n_qubits
        assert int(comparison["model"]["n_qubits"]) == n_qubits

        trained = {
            (int(item["degree"]), float(item["depolarizing_rate"])): item
            for item in training["records"]
        }
        compared = {
            (
                int(item["degree"]),
                float(item["depolarizing_rate"]),
                item["method"],
            ): item
            for item in comparison["records"]
        }
        for degree in expected_degrees:
            for rate in expected_rates:
                density = compared[(degree, rate, "density_matrix_training")]
                pre_qsp = compared[(degree, rate, "pre_qsp")]
                source = trained[(degree, rate)]
                assert density["seed_simulator"] == pre_qsp["seed_simulator"]
                assert np.allclose(
                    density["trainable_phases"],
                    source["trainable_phases"],
                    rtol=0.0,
                    atol=1.0e-14,
                )
                assert abs(
                    density["exact_density_evaluation"][
                        "energy_measurement_model"
                    ]
                    - source["exact_density_evaluation"][
                        "energy_measurement_model"
                    ]
                ) < 1.0e-12
                for record in (density, pre_qsp):
                    finite = record["finite_shot_evaluation"]
                    repetitions = int(finite["repetitions"])
                    valid = int(
                        finite.get("valid_energy_repetitions", repetitions)
                    )
                    invalid = int(finite.get("invalid_energy_repetitions", 0))
                    assert valid > 0 and valid + invalid == repetitions
                    assert int(finite["shots_per_basis"]) == 20000
                    assert int(finite["total_shots"]) == 200000

        for evaluation in ("exact_density", "finite_shot"):
            figure = (
                ROOT
                / "figures"
                / f"tfim_n{n_qubits}_pre_qsp_vs_density_"
                f"{evaluation}_noisy_only.png"
            )
            assert figure.is_file() and figure.stat().st_size > 0
    print("  PASS portable n=2/n=4 report snapshots and paired evaluation")


def _absolute_strings(value):
    if isinstance(value, str):
        return [value] if Path(value).expanduser().is_absolute() else []
    if isinstance(value, dict):
        return [
            item
            for nested in value.values()
            for item in _absolute_strings(nested)
        ]
    if isinstance(value, list):
        return [item for nested in value for item in _absolute_strings(nested)]
    return []


def _contains_key(value, target):
    if isinstance(value, dict):
        return target in value or any(
            _contains_key(nested, target) for nested in value.values()
        )
    if isinstance(value, list):
        return any(_contains_key(nested, target) for nested in value)
    return False


def check_short_training(model):
    training = {
        "optimizer": "adam",
        "iterations": 8,
        "learning_rate": 0.03,
        "final_learning_rate": 0.01,
        "gradient_clip": 5.0,
        "print_interval": 8,
        "checkpoint_interval": 8,
        "history_phase_interval": 0,
        "early_stopping": {"enabled": False},
        "polish": {
            "enabled": True,
            "optimizer": "lbfgs",
            "max_iterations": 10,
            "max_evaluations": 15,
            "max_passes": 2,
            "rescue_gradient_norm": 1.0e-30,
            "rescue_max_iterations": 2,
            "rescue_max_evaluations": 3,
            "learning_rate": 1.0,
            "line_search": "strong_wolfe",
        },
    }
    loss = {
        "type": "energy",
        "energy_source": "measurement_model",
        "include_measurement_basis_noise": True,
        "success_probability_floor": 0.05,
        "success_penalty_weight": 0.0,
    }
    initial = np.random.default_rng(7).uniform(-0.25, 0.25, 7)
    trainer = DirectDensityMatrixTrainer(model, training, loss, 1.0e-3)
    result = trainer.train(initial)
    initial_energy = result["history"][0]["metrics"]["energy_measurement_model"]
    best_energy = result["best"]["metrics"]["energy_measurement_model"]
    assert best_energy < initial_energy - 0.5
    assert result["polish"]["enabled"]
    assert result["polish"]["rescue_triggered"]
    assert len(result["polish"]["passes"]) == 2
    assert "trainable_phases" not in result["history"][0]
    assert "phi_su2" not in result["history"][0]
    assert result["training_shots"] == 0
    print(
        "  PASS direct-gradient training decreases noisy energy "
        f"({initial_energy:+.6f} -> {best_energy:+.6f})"
    )


if __name__ == "__main__":
    main()
