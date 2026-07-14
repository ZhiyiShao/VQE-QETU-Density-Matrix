from pathlib import Path
import argparse
import copy
import random

import numpy as np
import torch

from io_utils import (
    load_yaml,
    portable_metadata,
    portable_path,
    resolve_path,
    save_torch_atomic,
    timestamp,
    write_yaml_atomic,
)
from qetu_density import DifferentiableTFIMQETU, metrics_to_python
from qiskit_validation import finite_shot_evaluation, validate_torch_implementation
from trainer import DirectDensityMatrixTrainer, initialize_phases


_PRE_QSP_PAYLOAD_CACHE = {}


def main():
    parser = argparse.ArgumentParser(
        description="Train QETU phases with differentiable density matrices."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--depolarizing-rate", type=float)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--restarts", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-finite-shots", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    cfg = load_yaml(config_path)
    if args.depolarizing_rate is not None:
        cfg["noise"]["training_rate"] = float(args.depolarizing_rate)
        cfg["noise"]["evaluation_rates"] = [float(args.depolarizing_rate)]
    if args.iterations is not None:
        cfg["training"]["iterations"] = int(args.iterations)
    if args.restarts is not None:
        cfg["training"]["restarts"] = int(args.restarts)
    if args.no_finite_shots:
        cfg["validation"]["finite_shots"]["enabled"] = False
    result = run_experiment(
        cfg,
        config_path,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    print(f"Result: {result['result_file']}")


def run_experiment(cfg, config_path, overwrite=False, dry_run=False):
    cfg = copy.deepcopy(cfg)
    validate_config(cfg)
    seed = int(cfg["training"].get("seed", 7))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    output_dir = resolve_path(config_path, cfg["paths"]["output_dir"])
    result_path = output_dir / (
        "dry_run_result.yml" if dry_run else "training_result.yml"
    )
    if result_path.exists() and not overwrite:
        raise FileExistsError(
            f"{result_path} already exists. Use --overwrite to replace it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg)
    qetu_repo = resolve_path(config_path, cfg["paths"]["qetu_repo"])
    pre_qsp_record = load_pre_qsp_record(cfg, config_path, model)
    training_rate = float(cfg["noise"]["training_rate"])
    pre_qsp_baseline = evaluate_pre_qsp_baseline(
        model,
        pre_qsp_record,
        training_rate,
        cfg,
    )
    restarts = int(cfg["training"].get("restarts", 1))

    print(f"Experiment: {cfg['experiment']}")
    print(
        f"Model: TFIM n={model.n_system}, g={model.g:g}; "
        f"degree={model.degree}; symmetry={model.phase_symmetry}; "
        f"trainable phases={model.num_trainable_phases}"
    )
    print(
        "Training: exact PyTorch density matrix/autograd; "
        f"r_dplz={training_rate:.3g}; shots=0; restarts={restarts}"
    )
    finite_cfg = cfg["validation"]["finite_shots"]
    aer_enabled = bool(
        cfg["validation"]["aer_density_matrix"].get("enabled", True)
    )
    if bool(finite_cfg.get("enabled", True)):
        print(
            "Final validation: "
            + ("Aer density matrix and " if aer_enabled else "")
            + f"{finite_cfg['shots_per_basis']:,} shots/basis"
        )
    elif aer_enabled:
        print("Final validation: Aer density matrix; finite shots disabled")
    else:
        print("Final validation: exact PyTorch density matrix only")

    common = {
        "schema_version": 3,
        "experiment": cfg["experiment"],
        "status": "running",
        "started_at": timestamp(),
        "config_file": portable_path(config_path),
        "config": portable_metadata(cfg),
        "environment": model.summary(),
        "training_noise": noise_summary(cfg, training_rate),
        "pre_qsp_baseline": pre_qsp_baseline,
        "differentiation": {
            "engine": "PyTorch complex density matrix",
            "gradient": "reverse-mode autograd",
            "shots_during_training": 0,
            "fixed_block_compilation": "gate-wise noisy superoperator",
            "history_phase_interval": int(
                cfg["training"].get("history_phase_interval", 0)
            ),
        },
    }

    if dry_run:
        initialization_cfg, initialization_role = restart_initialization_config(
            cfg,
            0,
        )
        initial, initialization = initialize_phases(
            model,
            initialization_cfg,
            seed,
            pre_qsp_record,
        )
        initialization.update(initialization_role)
        with torch.no_grad():
            metrics = model.simulate(
                initial,
                training_rate,
                measurement_basis_noise=bool(
                    cfg["loss"].get("include_measurement_basis_noise", True)
                ),
            )
        validation = validate_torch_implementation(
            model,
            initial,
            training_rate,
            qetu_repo,
            atol_noiseless=float(
                cfg["validation"]["aer_density_matrix"].get(
                    "atol_noiseless", 1.0e-10
                )
            ),
            atol_noisy=float(
                cfg["validation"]["aer_density_matrix"].get(
                    "atol_noisy", 2.0e-9
                )
            ),
        )
        payload = dict(common)
        payload.update(
            {
                "status": "dry_run_complete",
                "finished_at": timestamp(),
                "initialization": initialization,
                "metrics": metrics_to_python(metrics),
                "phi_su2": metrics["phi_su2"].detach().cpu().tolist(),
                "cross_validation": validation,
            }
        )
        write_yaml_atomic(result_path, payload)
        return {"payload": payload, "result_file": str(result_path)}

    restart_results = []
    best_restart = None
    for restart_index in range(restarts):
        restart_seed = seed + restart_index
        initialization_cfg, initialization_role = restart_initialization_config(
            cfg,
            restart_index,
        )
        initial, initialization = initialize_phases(
            model,
            initialization_cfg,
            restart_seed,
            pre_qsp_record,
        )
        initialization.update(initialization_role)
        trainer = DirectDensityMatrixTrainer(
            model,
            cfg["training"],
            cfg["loss"],
            training_rate,
        )

        def checkpoint_callback(partial):
            checkpoint = {
                "restart_index": partial["restart_index"],
                "iteration": partial["iteration"],
                "current_trainable_phases": partial[
                    "current_trainable_phases"
                ],
                "optimizer_state_dict": partial["optimizer_state_dict"],
                "best": partial["best"],
                "config": cfg,
            }
            save_torch_atomic(output_dir / "checkpoint.pt", checkpoint)

        print(f"Restart {restart_index + 1}/{restarts} (seed={restart_seed})")
        training_result = trainer.train(
            initial,
            restart_index=restart_index,
            checkpoint_callback=checkpoint_callback,
        )
        training_result["initialization"] = initialization
        restart_results.append(training_result)
        if (
            best_restart is None
            or training_result["best"]["loss"] < best_restart["best"]["loss"]
        ):
            best_restart = training_result

    best_trainable = np.asarray(
        best_restart["best"]["trainable_phases"],
        dtype=float,
    )
    with torch.no_grad():
        ideal_phase_metrics = metrics_to_python(
            model.simulate(
                best_trainable,
                0.0,
                measurement_basis_noise=False,
            )
        )
    ideal_phase_metrics["energy_absolute_error"] = abs(
        ideal_phase_metrics["energy_state"]
        - model.ground_data["ground_energy"]
    )
    exact_evaluations = evaluate_noise_rates(model, best_trainable, cfg)
    training_rate_evaluation = min(
        exact_evaluations,
        key=lambda item: abs(
            float(item["depolarizing_rate"]) - training_rate
        ),
    )
    improvement = baseline_improvement(
        pre_qsp_baseline,
        training_rate_evaluation,
    )
    aer_cfg = cfg["validation"]["aer_density_matrix"]
    cross_validation = None
    if bool(aer_cfg.get("enabled", True)):
        cross_validation = validate_torch_implementation(
            model,
            best_trainable,
            training_rate,
            qetu_repo,
            atol_noiseless=float(aer_cfg.get("atol_noiseless", 1.0e-10)),
            atol_noisy=float(aer_cfg.get("atol_noisy", 2.0e-9)),
        )
        if bool(aer_cfg.get("fail_on_mismatch", True)) and not cross_validation[
            "passed"
        ]:
            raise RuntimeError(
                "PyTorch/QETU/Aer cross-validation failed; see checkpoint output."
            )

    finite_cfg = cfg["validation"]["finite_shots"]
    finite_evaluations = []
    if bool(finite_cfg.get("enabled", True)):
        for rate in evaluation_rates(cfg):
            print(
                f"Finite-shot validation r={rate:.0e}: "
                f"{finite_cfg['repetitions']} x "
                f"{finite_cfg['shots_per_basis']:,} shots/basis"
            )
            finite_evaluations.append(
                finite_shot_evaluation(
                    model,
                    best_trainable,
                    qetu_repo,
                    rate,
                    int(finite_cfg["shots_per_basis"]),
                    int(finite_cfg["repetitions"]),
                    int(finite_cfg.get("seed_simulator", 200001)),
                )
            )

    total_objective_evaluations = sum(
        int(item["objective_evaluations"]) for item in restart_results
    )
    payload = dict(common)
    payload.update(
        {
            "status": "complete",
            "finished_at": timestamp(),
            "restarts": restart_results,
            "selected_restart": int(best_restart["restart_index"]),
            "best": best_restart["best"],
            "ideal_phase_quality": ideal_phase_metrics,
            "exact_density_evaluations": exact_evaluations,
            "improvement_over_pre_qsp": improvement,
            "cross_validation": cross_validation,
            "finite_shot_evaluations": finite_evaluations,
            "resources": {
                "training_objective_evaluations": total_objective_evaluations,
                "training_shots": 0,
                "final_validation_shots": int(
                    sum(item["total_shots"] for item in finite_evaluations)
                ),
                "wall_time_seconds": float(
                    sum(item["wall_time_seconds"] for item in restart_results)
                ),
            },
        }
    )
    write_yaml_atomic(result_path, payload)
    save_torch_atomic(
        output_dir / "best_phases.pt",
        {
            "trainable_phases": torch.as_tensor(best_trainable),
            "phi_su2": torch.as_tensor(best_restart["best"]["phi_su2"]),
            "best": best_restart["best"],
            "config": cfg,
        },
    )
    print_best_summary(payload)
    return {"payload": payload, "result_file": str(result_path)}


def build_model(cfg):
    model_cfg = cfg["model"]
    oracle_cfg = cfg["oracle"]
    trotter_cfg = oracle_cfg["trotter"]
    return DifferentiableTFIMQETU(
        n_qubits=int(model_cfg["n_qubits"]),
        g=float(model_cfg["g"]),
        pbc=bool(model_cfg.get("pbc", False)),
        eta=float(cfg["spectral_scaling"]["eta"]),
        tau=float(oracle_cfg["tau"]),
        trotter_steps=int(trotter_cfg["steps"]),
        trotter_order=int(trotter_cfg["order"]),
        degree=int(cfg["ansatz"]["degree"]),
        phase_symmetry=str(cfg["ansatz"]["symmetry"]),
        initial_bitstring=str(cfg["initial_state"]["bitstring"]),
        one_qubit_rate_factor=float(cfg["noise"].get("one_qubit_rate_factor", 0.1)),
        dtype=torch.complex128,
        device=str(cfg["training"].get("device", "cpu")),
    )


def evaluate_noise_rates(model, trainable_phases, cfg):
    evaluations = []
    for rate in evaluation_rates(cfg):
        with torch.no_grad():
            metrics = model.simulate(
                trainable_phases,
                rate,
                measurement_basis_noise=bool(
                    cfg["loss"].get("include_measurement_basis_noise", True)
                ),
            )
        values = metrics_to_python(metrics)
        values.update(
            {
                "depolarizing_rate": float(rate),
                "energy_state_absolute_error": abs(
                    values["energy_state"] - model.ground_data["ground_energy"]
                ),
                "energy_measurement_absolute_error": abs(
                    values["energy_measurement_model"]
                    - model.ground_data["ground_energy"]
                ),
            }
        )
        evaluations.append(values)
    return evaluations


def evaluation_rates(cfg):
    rates = cfg["noise"].get(
        "evaluation_rates",
        [cfg["noise"]["training_rate"]],
    )
    values = [float(rate) for rate in rates]
    if not values or any(
        not np.isfinite(rate) or rate < 0.0 or rate > 1.0
        for rate in values
    ):
        raise ValueError("noise.evaluation_rates must lie in [0, 1].")
    return values


def evaluate_pre_qsp_baseline(model, record, rate, cfg):
    if record is None:
        return None
    phases, _ = initialize_phases(
        model,
        {"method": "pre_qsp"},
        seed=0,
        pre_qsp_record=record,
    )
    with torch.no_grad():
        metrics = metrics_to_python(
            model.simulate(
                phases,
                rate,
                measurement_basis_noise=bool(
                    cfg["loss"].get("include_measurement_basis_noise", True)
                ),
            )
        )
    exact = float(model.ground_data["ground_energy"])
    metrics.update(
        {
            "depolarizing_rate": float(rate),
            "trainable_phases": phases.tolist(),
            "energy_measurement_absolute_error": abs(
                metrics["energy_measurement_model"] - exact
            ),
            "energy_state_absolute_error": abs(
                metrics["energy_state"] - exact
            ),
        }
    )
    return metrics


def baseline_improvement(baseline, trained):
    if baseline is None:
        return None
    baseline_error = float(baseline["energy_measurement_absolute_error"])
    trained_error = float(trained["energy_measurement_absolute_error"])
    reduction = baseline_error - trained_error
    return {
        "baseline_energy_absolute_error": baseline_error,
        "trained_energy_absolute_error": trained_error,
        "absolute_error_reduction": reduction,
        "relative_error_reduction": (
            reduction / baseline_error if baseline_error > 0.0 else 0.0
        ),
        "success_probability_change": float(
            trained["success_probability_ancilla0"]
            - baseline["success_probability_ancilla0"]
        ),
    }


def load_pre_qsp_record(cfg, config_path, model):
    initialization_requires_pre_qsp = str(
        cfg["initialization"].get("method")
    ) in (
        "pre_qsp",
        "pre_qsp_plus_random",
    )
    path_value = cfg["paths"].get("pre_qsp_results")
    if path_value is None:
        if initialization_requires_pre_qsp:
            raise ValueError(
                "paths.pre_qsp_results is required for pre_qsp initialization."
            )
        return None
    path = resolve_path(config_path, path_value)
    payload = load_pre_qsp_payload(path)
    model_cfg = payload.get("model", {})
    if (
        int(model_cfg.get("n_qubits", -1)) != model.n_system
        or not np.isclose(float(model_cfg.get("g", np.nan)), model.g)
        or bool(model_cfg.get("pbc", False)) != model.pbc
    ):
        raise ValueError("The pre-QSP result uses a different TFIM model.")
    if payload.get("initial_state") != cfg["initial_state"]:
        raise ValueError("The pre-QSP result uses a different initial state.")
    source_scaling = payload.get("spectral_scaling", {})
    if (
        source_scaling.get("method") != "exact_spectrum"
        or not np.isclose(
            float(source_scaling.get("eta", np.nan)),
            model.scaling.eta,
            rtol=0.0,
            atol=1.0e-12,
        )
    ):
        raise ValueError("The pre-QSP result uses different spectral scaling.")
    source_oracle = payload.get("oracle", {})
    if (
        source_oracle.get("implementation")
        != cfg["oracle"]["implementation"]
        or not np.isclose(
            float(source_oracle.get("tau", np.nan)),
            model.tau,
            rtol=0.0,
            atol=1.0e-12,
        )
        or int(source_oracle.get("trotter_steps", -1))
        != model.trotter_steps
        or int(source_oracle.get("trotter_order", -1))
        != model.trotter_order
    ):
        raise ValueError("The pre-QSP result uses a different QETU oracle.")
    for record in payload.get("records", []):
        if int(record.get("degree", -1)) == model.degree and record.get("status") == "ok":
            full = np.asarray(
                record["phases"]["phi_su2_for_qetu_api"],
                dtype=float,
            )
            if full.size != model.degree + 1 or not np.all(np.isfinite(full)):
                raise ValueError("The pre-QSP phase sequence has the wrong form.")
            symmetry_error = np.max(
                np.abs((full - full[::-1] + np.pi) % (2.0 * np.pi) - np.pi)
            )
            if symmetry_error > 1.0e-9:
                raise ValueError("The pre-QSP phase sequence is not palindromic.")
            return record
    raise ValueError(f"No successful degree-{model.degree} record in {path}.")


def load_pre_qsp_payload(path):
    """Load a pre-QSP scan once per process unless the source file changes."""
    path = Path(path).resolve()
    stat = path.stat()
    signature = (int(stat.st_mtime_ns), int(stat.st_size))
    key = str(path)
    cached = _PRE_QSP_PAYLOAD_CACHE.get(key)
    if cached is not None and cached[0] == signature:
        return cached[1]
    payload = load_yaml(path)
    _PRE_QSP_PAYLOAD_CACHE[key] = (signature, payload)
    return payload


def noise_summary(cfg, rate):
    return {
        "model": "qetu_gatewise_depolarizing",
        "depolarizing_rate": float(rate),
        "one_qubit_rate_factor": float(cfg["noise"].get("one_qubit_rate_factor", 0.1)),
        "one_qubit_error_rate": float(
            rate * float(cfg["noise"].get("one_qubit_rate_factor", 0.1))
        ),
        "two_qubit_error_rate": float(rate),
        "readout_error_enabled": False,
    }


def restart_initialization_config(cfg, restart_index):
    config = copy.deepcopy(cfg["initialization"])
    requested = str(config.get("method", "random_uniform"))
    role = "configured"
    if requested == "pre_qsp_plus_random":
        if int(restart_index) == 0:
            config["method"] = "pre_qsp"
            role = "pre_qsp_warm_start"
        else:
            config["method"] = "random_uniform"
            role = "random_restart"
    return config, {
        "requested_method": requested,
        "restart_role": role,
    }


def validate_config(cfg):
    required = (
        "experiment",
        "paths",
        "model",
        "initial_state",
        "spectral_scaling",
        "oracle",
        "ansatz",
        "initialization",
        "loss",
        "noise",
        "training",
        "validation",
    )
    for section in required:
        if section not in cfg:
            raise ValueError(f"Missing config section: {section}")
    if cfg["model"].get("name") != "tfim":
        raise ValueError("This benchmark supports model.name=tfim.")
    if cfg["spectral_scaling"].get("method") != "exact_spectrum":
        raise ValueError("spectral_scaling.method must be exact_spectrum.")
    if cfg["oracle"].get("implementation") != "qetu_main_tfim_control_free":
        raise ValueError("oracle.implementation must be qetu_main_tfim_control_free.")
    if cfg["ansatz"].get("phase_convention") != "qetu_su2":
        raise ValueError("ansatz.phase_convention must be qetu_su2.")
    if cfg["ansatz"].get("symmetry") not in ("palindrome", "full"):
        raise ValueError("ansatz.symmetry must be palindrome or full.")
    initialization_method = str(cfg["initialization"].get("method", ""))
    supported_initializations = (
        "random_uniform",
        "zeros",
        "pi_quarter_endpoints",
        "pre_qsp",
        "pre_qsp_plus_random",
    )
    if initialization_method not in supported_initializations:
        raise ValueError(
            "Unsupported initialization.method: " + initialization_method
        )
    if cfg["noise"].get("model") != "qetu_gatewise_depolarizing":
        raise ValueError("noise.model must be qetu_gatewise_depolarizing.")
    evaluation_rates(cfg)
    training_rate = float(cfg["noise"]["training_rate"])
    if not np.isfinite(training_rate) or not 0.0 <= training_rate <= 1.0:
        raise ValueError("noise.training_rate must lie in [0, 1].")
    one_qubit_factor = float(cfg["noise"].get("one_qubit_rate_factor", 0.1))
    if not np.isfinite(one_qubit_factor) or not 0.0 < one_qubit_factor <= 1.0:
        raise ValueError("noise.one_qubit_rate_factor must lie in (0, 1].")
    if int(cfg["training"].get("restarts", 1)) <= 0:
        raise ValueError("training.restarts must be positive.")
    if (
        initialization_method == "pre_qsp_plus_random"
        and int(cfg["training"].get("restarts", 1)) < 2
    ):
        raise ValueError("pre_qsp_plus_random requires at least two restarts.")
    if int(cfg["training"]["iterations"]) <= 0:
        raise ValueError("training.iterations must be positive.")
    finite = cfg["validation"]["finite_shots"]
    if int(finite["shots_per_basis"]) <= 0 or int(finite["repetitions"]) <= 0:
        raise ValueError("Finite-shot counts must be positive.")


def print_best_summary(payload):
    best = payload["best"]
    metrics = best["metrics"]
    exact = payload["environment"]["model"]["exact_ground_energy"]
    print("Best phase solution")
    print(f"  energy (measurement model): {metrics['energy_measurement_model']:+.12f}")
    print(f"  energy (postselected state): {metrics['energy_state']:+.12f}")
    print(f"  exact ground energy:         {exact:+.12f}")
    print(f"  state energy abs error:      {abs(metrics['energy_state'] - exact):.3e}")
    print(f"  ground-space fidelity:       {metrics['fidelity_ground_space']:.9f}")
    print(f"  success probability:         {metrics['success_probability_ancilla0']:.6f}")
    ideal = payload["ideal_phase_quality"]
    print(f"  ideal-circuit fidelity:       {ideal['fidelity_ground_space']:.9f}")
    print(f"  ideal-circuit energy error:   {ideal['energy_absolute_error']:.3e}")
    print(f"  training shots:              {payload['resources']['training_shots']}")
    gradient_norm = best.get("gradient_norm")
    if gradient_norm is not None:
        print(f"  final gradient norm:         {gradient_norm:.3e}")
    selected = payload["restarts"][int(payload["selected_restart"])]
    polish = selected.get("polish", {})
    if polish.get("enabled"):
        print(f"  L-BFGS passes:               {len(polish.get('passes', []))}")
    improvement = payload.get("improvement_over_pre_qsp")
    if improvement is not None:
        print(
            "  error reduction vs pre-QSP:  "
            f"{improvement['relative_error_reduction']:.2%}"
        )


if __name__ == "__main__":
    main()
