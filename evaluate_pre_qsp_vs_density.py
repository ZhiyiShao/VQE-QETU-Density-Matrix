from pathlib import Path
import argparse
import math
import platform
import subprocess
import sys
import traceback

import numpy as np
import torch

from io_utils import (
    load_yaml,
    portable_metadata,
    portable_path,
    resolve_path,
    timestamp,
    write_yaml_atomic,
)
from qetu_density import DifferentiableTFIMQETU, metrics_to_python
from qiskit_validation import finite_shot_evaluation


METHODS = ("pre_qsp", "density_matrix_training")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Reevaluate saved pre-QSP and density-trained phases with one "
            "shared finite-shot protocol."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--degrees", type=int, nargs="+")
    parser.add_argument("--rates", type=float, nargs="+")
    parser.add_argument("--shots", type=int)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    cfg = load_yaml(config_path)
    if args.shots is not None:
        cfg["evaluation"]["shots_per_basis"] = int(args.shots)
    if args.repetitions is not None:
        cfg["evaluation"]["repetitions"] = int(args.repetitions)
    validate_config(cfg)

    degrees = (
        validate_degrees(args.degrees)
        if args.degrees is not None
        else degree_values(cfg["comparison"]["degrees"])
    )
    rates = (
        validate_rates(args.rates)
        if args.rates is not None
        else validate_rates(cfg["comparison"]["noise_rates"])
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else resolve_path(config_path, cfg["paths"]["output_dir"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    shots = int(cfg["evaluation"]["shots_per_basis"])
    result_path = output_dir / f"comparison_shot{shots}.yml"

    pre_qsp = load_yaml(resolve_path(config_path, cfg["paths"]["pre_qsp_result"]))
    density = load_yaml(resolve_path(config_path, cfg["paths"]["density_result"]))
    validate_sources(cfg, pre_qsp, density, degrees, rates)
    reference_model = build_model(cfg, degrees[0])
    summary = initialize_summary(
        result_path,
        cfg,
        config_path,
        degrees,
        rates,
        reference_model,
        overwrite=args.overwrite,
        resume=args.resume,
    )
    write_yaml_atomic(result_path, summary)
    try:
        aer_version = verify_aer_runtime()
    except RuntimeError as error:
        summary["status"] = "failed"
        summary["updated_at"] = timestamp()
        summary["error"] = str(error)
        write_yaml_atomic(result_path, summary)
        raise
    summary.pop("error", None)
    summary["runtime"] = {
        "python_version": platform.python_version(),
        "qiskit_aer_version": aer_version,
    }
    write_yaml_atomic(result_path, summary)
    qetu_repo = resolve_path(config_path, cfg["paths"]["qetu_repo"])
    repetitions = int(cfg["evaluation"]["repetitions"])
    base_seed = int(cfg["evaluation"]["seed_simulator"])

    print("Paired pre-QSP vs density-matrix reevaluation")
    print(f"  degrees: {degrees}")
    print(f"  r_dplz: {rates}")
    print(f"  shots: {shots:,}/basis x {repetitions} repetitions")
    print("  phase training: not repeated; saved phases are reused")
    print("  simulator seeds: paired between methods at each scan point")

    for degree in degrees:
        model = build_model(cfg, degree)
        pre_phases = pre_qsp_trainable_phases(pre_qsp, model)
        for rate_index, rate in enumerate(rates):
            density_phases = density_trainable_phases(density, model, rate)
            seed = base_seed + 1000 * int(degree) + 20 * rate_index
            for method, phases in (
                ("pre_qsp", pre_phases),
                ("density_matrix_training", density_phases),
            ):
                if args.resume and find_record(
                    summary["records"], degree, rate, method, status="ok"
                ):
                    print(
                        f"Skipping {method}: degree={degree}, r={rate:.0e}"
                    )
                    continue
                print(f"\n=== {method}: degree={degree}, r={rate:.0e} ===")
                try:
                    with torch.no_grad():
                        exact = metrics_to_python(
                            model.simulate(
                                phases,
                                rate,
                                measurement_basis_noise=True,
                            )
                        )
                    exact_ground = float(model.ground_data["ground_energy"])
                    exact["energy_state_absolute_error"] = abs(
                        exact["energy_state"] - exact_ground
                    )
                    exact["energy_measurement_absolute_error"] = abs(
                        exact["energy_measurement_model"] - exact_ground
                    )
                    finite = finite_shot_evaluation(
                        model=model,
                        trainable_phases=phases,
                        qetu_repo=qetu_repo,
                        depolarizing_rate=rate,
                        shots_per_basis=shots,
                        repetitions=repetitions,
                        seed_simulator=seed,
                    )
                    record = {
                        "degree": int(degree),
                        "depolarizing_rate": float(rate),
                        "method": method,
                        "status": "ok",
                        "seed_simulator": int(seed),
                        "trainable_phases": np.asarray(
                            phases, dtype=float
                        ).tolist(),
                        "exact_density_evaluation": exact,
                        "finite_shot_evaluation": finite,
                    }
                    print(
                        f"E_exact={exact['energy_measurement_model']:+.9f}; "
                        f"E_shot={finite['energy_mean']:+.9f} +/- "
                        f"{finite['energy_standard_deviation']:.3e}; "
                        f"p0={exact['success_probability_ancilla0']:.4f}"
                    )
                except KeyboardInterrupt:
                    summary["status"] = "interrupted"
                    summary["updated_at"] = timestamp()
                    write_yaml_atomic(result_path, summary)
                    raise
                except Exception as error:
                    record = {
                        "degree": int(degree),
                        "depolarizing_rate": float(rate),
                        "method": method,
                        "status": "failed",
                        "error": " ".join(str(error).split()),
                        "traceback": traceback.format_exc(),
                    }
                    print(f"FAILED: {record['error']}")
                    if args.fail_fast:
                        replace_record(summary["records"], record)
                        summary["status"] = "failed"
                        write_yaml_atomic(result_path, summary)
                        raise

                replace_record(summary["records"], record)
                summary["records"].sort(
                    key=lambda item: (
                        int(item["degree"]),
                        float(item["depolarizing_rate"]),
                        str(item["method"]),
                    )
                )
                summary["updated_at"] = timestamp()
                write_yaml_atomic(result_path, summary)

    failures = [item for item in summary["records"] if item["status"] != "ok"]
    summary["status"] = "complete" if not failures else "complete_with_failures"
    summary["finished_at"] = timestamp()
    summary["resources"] = {
        "phase_training_repeated": False,
        "total_finite_shots": int(
            sum(
                item["finite_shot_evaluation"]["total_shots"]
                for item in summary["records"]
                if item["status"] == "ok"
            )
        ),
    }
    write_yaml_atomic(result_path, summary)
    print(f"\nComparison status: {summary['status']}")
    print(f"Comparison result: {result_path}")
    if failures:
        raise SystemExit(1)


def build_model(cfg, degree):
    model_cfg = cfg["model"]
    oracle_cfg = cfg["oracle"]
    return DifferentiableTFIMQETU(
        n_qubits=int(model_cfg["n_qubits"]),
        g=float(model_cfg["g"]),
        pbc=bool(model_cfg.get("pbc", False)),
        eta=float(cfg["spectral_scaling"]["eta"]),
        tau=float(oracle_cfg["tau"]),
        trotter_steps=int(oracle_cfg["trotter"]["steps"]),
        trotter_order=int(oracle_cfg["trotter"]["order"]),
        degree=int(degree),
        phase_symmetry=str(cfg["ansatz"]["symmetry"]),
        initial_bitstring=str(cfg["initial_state"]["bitstring"]),
        one_qubit_rate_factor=float(
            cfg["evaluation"].get("one_qubit_rate_factor", 0.1)
        ),
    )


def pre_qsp_trainable_phases(payload, model):
    degree = model.degree
    record = next(
        (
            item
            for item in payload["records"]
            if int(item["degree"]) == int(degree)
            and item.get("status") == "ok"
        ),
        None,
    )
    if record is None:
        raise ValueError(f"No successful pre-QSP degree-{degree} record.")
    full = np.asarray(
        record["phases"]["phi_su2_for_qetu_api"],
        dtype=float,
    )
    validate_full_phases(full, degree, "pre-QSP")
    if model.phase_symmetry == "palindrome":
        return full[: degree // 2 + 1]
    return full


def density_trainable_phases(payload, model, rate):
    degree = model.degree
    record = next(
        (
            item
            for item in payload["records"]
            if int(item["degree"]) == int(degree)
            and abs(float(item["depolarizing_rate"]) - float(rate)) < 1.0e-15
            and item.get("status") == "ok"
        ),
        None,
    )
    if record is None:
        raise ValueError(
            f"No successful density degree-{degree}, r={rate:.0e} record."
        )
    saved_phases = record.get("trainable_phases")
    if saved_phases is None:
        saved_phases = record.get("independent_phases")
    if saved_phases is None:
        raise ValueError("Saved density record does not contain phases.")
    phases = np.asarray(saved_phases, dtype=float)
    if phases.size != model.num_trainable_phases or not np.all(np.isfinite(phases)):
        raise ValueError("Saved density-trained phases have the wrong form.")
    return phases


def validate_full_phases(phases, degree, label):
    if phases.size != int(degree) + 1 or not np.all(np.isfinite(phases)):
        raise ValueError(f"{label} phase sequence has the wrong form.")
    symmetry = np.max(
        np.abs((phases - phases[::-1] + math.pi) % (2.0 * math.pi) - math.pi)
    )
    if symmetry > 1.0e-9:
        raise ValueError(f"{label} phase sequence is not palindromic.")


def validate_sources(cfg, pre_qsp, density, degrees, rates):
    for source, label in ((pre_qsp, "pre-QSP"), (density, "density")):
        model = source.get("model", source.get("config", {}).get("model", {}))
        expected = cfg["model"]
        if (
            model.get("name") != expected["name"]
            or int(model.get("n_qubits", -1)) != int(expected["n_qubits"])
            or not np.isclose(float(model.get("g", np.nan)), float(expected["g"]))
            or bool(model.get("pbc", False)) != bool(expected.get("pbc", False))
        ):
            raise ValueError(f"{label} source uses a different model.")
    density_symmetry = density.get("config", {}).get("ansatz", {}).get(
        "symmetry"
    )
    if density_symmetry != cfg["ansatz"]["symmetry"]:
        raise ValueError(
            "The density source and comparison use different phase symmetry."
        )
    for degree in degrees:
        model = build_model(cfg, degree)
        pre_qsp_trainable_phases(pre_qsp, model)
        for rate in rates:
            density_trainable_phases(density, model, rate)


def initialize_summary(
    path,
    cfg,
    config_path,
    degrees,
    rates,
    model,
    overwrite,
    resume,
):
    if path.exists() and resume:
        summary = load_yaml(path)
        if summary.get("config") != portable_metadata(cfg):
            raise ValueError("Evaluation config changed; use --overwrite.")
        if summary.get("degrees") != degrees or summary.get("noise_rates") != rates:
            raise ValueError("Saved comparison uses different scan points.")
        summary["status"] = "running"
        return summary
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists. Use --resume or --overwrite.")
    return {
        "schema_version": 2,
        "benchmark": cfg["benchmark"],
        "status": "running",
        "started_at": timestamp(),
        "updated_at": timestamp(),
        "config_file": portable_path(config_path),
        "config": portable_metadata(cfg),
        "model": model.summary()["model"],
        "degrees": degrees,
        "noise_rates": rates,
        "methods": list(METHODS),
        "comparison_protocol": {
            "same_shots": True,
            "same_repetitions": True,
            "paired_simulator_seeds": True,
            "saved_phases_reused": True,
            "phase_training_repeated": False,
        },
        "records": [],
    }


def validate_config(cfg):
    required = (
        "benchmark",
        "paths",
        "model",
        "initial_state",
        "spectral_scaling",
        "oracle",
        "ansatz",
        "comparison",
        "evaluation",
    )
    for section in required:
        if section not in cfg:
            raise ValueError(f"Missing config section: {section}")
    if cfg["model"].get("name") != "tfim":
        raise ValueError("This comparison supports TFIM only.")
    if cfg["spectral_scaling"].get("method") != "exact_spectrum":
        raise ValueError("spectral_scaling.method must be exact_spectrum.")
    if cfg["oracle"].get("implementation") != "qetu_main_tfim_control_free":
        raise ValueError("Unexpected QETU oracle implementation.")
    if cfg["ansatz"].get("symmetry") not in ("palindrome", "full"):
        raise ValueError("ansatz.symmetry must be palindrome or full.")
    shots = int(cfg["evaluation"]["shots_per_basis"])
    repetitions = int(cfg["evaluation"]["repetitions"])
    if shots <= 0 or repetitions <= 0:
        raise ValueError("shots_per_basis and repetitions must be positive.")
    if not bool(cfg["evaluation"].get("paired_simulator_seeds", True)):
        raise ValueError("This benchmark requires paired_simulator_seeds=true.")


def verify_aer_runtime():
    """Check Aer in a child process so a native import crash is reportable."""
    command = [
        sys.executable,
        "-c",
        "import qiskit_aer; print(qiskit_aer.__version__)",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("Qiskit Aer runtime check timed out.") from error

    if completed.returncode != 0:
        details = " ".join(
            (completed.stderr or completed.stdout or "").split()
        )
        if details:
            details = f" Runtime output: {details[-500:]}"
        raise RuntimeError(
            "Qiskit Aer cannot start in this environment. This project pins "
            "the known-compatible Qiskit 0.39.5 / Aer 0.11.2 / "
            "SciPy 1.10.1 combination; run `python -m pip install "
            "scipy==1.10.1` and then resume the comparison."
            f"{details}"
        )
    return completed.stdout.strip().splitlines()[-1]


def degree_values(cfg):
    return validate_degrees(
        list(range(int(cfg["start"]), int(cfg["stop"]) + 1, int(cfg["step"])))
    )


def validate_degrees(values):
    if not values:
        raise ValueError("At least one degree is required.")
    degrees = sorted(set(int(value) for value in values))
    if any(degree < 2 or degree % 2 for degree in degrees):
        raise ValueError("All degrees must be positive even integers.")
    return degrees


def validate_rates(values):
    if not values:
        raise ValueError("At least one noise rate is required.")
    rates = sorted(set(float(value) for value in values))
    if any(
        not math.isfinite(rate) or rate < 0.0 or rate > 1.0
        for rate in rates
    ):
        raise ValueError("Noise rates must lie in [0, 1].")
    return rates


def find_record(records, degree, rate, method, status=None):
    for record in records:
        if (
            int(record["degree"]) == int(degree)
            and abs(float(record["depolarizing_rate"]) - float(rate)) < 1.0e-15
            and record["method"] == method
            and (status is None or record.get("status") == status)
        ):
            return record
    return None


def replace_record(records, new_record):
    records[:] = [
        record
        for record in records
        if not (
            int(record["degree"]) == int(new_record["degree"])
            and abs(
                float(record["depolarizing_rate"])
                - float(new_record["depolarizing_rate"])
            )
            < 1.0e-15
            and record["method"] == new_record["method"]
        )
    ]
    records.append(new_record)


if __name__ == "__main__":
    main()
