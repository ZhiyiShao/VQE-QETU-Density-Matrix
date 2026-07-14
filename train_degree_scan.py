from pathlib import Path
import argparse
import copy
import math
import traceback

from io_utils import (
    load_yaml,
    portable_metadata,
    portable_path,
    resolve_path,
    timestamp,
    write_yaml_atomic,
)
from train_density_matrix import run_experiment, validate_config


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train QETU phases over degree and noise rate with a pure "
            "density-matrix energy loss."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--no-finite-shots", action="store_true")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--restarts", type=int)
    parser.add_argument("--degrees", type=int, nargs="+")
    parser.add_argument("--rates", type=float, nargs="+")
    parser.add_argument(
        "--phase-symmetry",
        choices=("palindrome", "full"),
        help="Override ansatz.symmetry for a parameter-count ablation.",
    )
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    base = load_yaml(config_path)
    if args.iterations is not None:
        base["training"]["iterations"] = int(args.iterations)
    if args.restarts is not None:
        base["training"]["restarts"] = int(args.restarts)
    if args.no_finite_shots:
        base["validation"]["finite_shots"]["enabled"] = False
    if args.phase_symmetry is not None:
        base["ansatz"]["symmetry"] = args.phase_symmetry
    validate_scan_config(base)

    degrees = (
        _validate_degrees(args.degrees)
        if args.degrees is not None
        else degree_values(base["degree_scan"])
    )
    rates = (
        _validate_rates(args.rates)
        if args.rates is not None
        else _validate_rates(base["noise"]["sweep_rates"])
    )
    output_root = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else resolve_path(config_path, base["paths"]["output_dir"])
    )
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "degree_scan_result.yml"
    summary = initialize_summary(
        summary_path,
        base,
        config_path,
        degrees,
        rates,
        overwrite=args.overwrite,
        resume=args.resume,
    )

    print("Differentiable density-matrix degree scan")
    print(f"  degrees: {degrees}")
    print(f"  r_dplz: {rates}")
    print(f"  initialization: {base['initialization']['method']}")
    print(f"  phase symmetry: {base['ansatz']['symmetry']}")
    print("  loss: normalized energy only; success penalty = 0")
    print(
        "  final evaluation: "
        + (
            f"{base['validation']['finite_shots']['shots_per_basis']:,} "
            f"shots/basis x {base['validation']['finite_shots']['repetitions']}"
            if base["validation"]["finite_shots"]["enabled"]
            else "exact density matrix only"
        )
    )

    for degree in degrees:
        degree_seed = int(base["training"].get("seed", 7)) + int(degree)
        for rate in rates:
            if args.resume and find_record(summary["records"], degree, rate, "ok"):
                print(f"Skipping completed degree={degree}, r={rate:.0e}")
                continue
            run_cfg = make_run_config(
                base,
                output_root,
                degree,
                rate,
                degree_seed,
            )
            result_path = resolve_path(
                config_path,
                run_cfg["paths"]["output_dir"],
            ) / "training_result.yml"
            print(f"\n=== degree={degree}, r_dplz={rate:.0e} ===")
            try:
                if args.resume and result_path.exists():
                    payload = load_yaml(result_path)
                    if payload.get("status") != "complete":
                        raise ValueError(f"Incomplete saved run: {result_path}")
                    result_file = str(result_path)
                else:
                    result = run_experiment(
                        run_cfg,
                        config_path,
                        overwrite=args.overwrite,
                        dry_run=False,
                    )
                    payload = result["payload"]
                    result_file = result["result_file"]
                record = summarize_run(
                    payload,
                    result_file,
                    degree,
                    rate,
                    degree_seed,
                )
            except KeyboardInterrupt:
                summary["status"] = "interrupted"
                summary["updated_at"] = timestamp()
                write_yaml_atomic(summary_path, summary)
                raise
            except Exception as error:
                record = {
                    "degree": int(degree),
                    "depolarizing_rate": float(rate),
                    "status": "failed",
                    "error": " ".join(str(error).split()),
                    "traceback": traceback.format_exc(),
                }
                print(f"FAILED: {record['error']}")
                if args.fail_fast:
                    replace_record(summary["records"], record)
                    summary["status"] = "failed"
                    write_yaml_atomic(summary_path, summary)
                    raise

            replace_record(summary["records"], record)
            summary["records"].sort(
                key=lambda item: (
                    int(item["degree"]),
                    float(item["depolarizing_rate"]),
                )
            )
            summary["updated_at"] = timestamp()
            write_yaml_atomic(summary_path, summary)

    failures = [item for item in summary["records"] if item["status"] != "ok"]
    summary["status"] = "complete" if not failures else "complete_with_failures"
    summary["finished_at"] = timestamp()
    write_yaml_atomic(summary_path, summary)
    print(f"\nScan status: {summary['status']}")
    print(f"Scan result: {summary_path}")


def make_run_config(base, output_root, degree, rate, seed):
    cfg = copy.deepcopy(base)
    tag = "ideal" if rate == 0.0 else f"rdplz_{rate:.0e}"
    cfg["experiment"] = (
        f"{base['experiment']}_degree{int(degree)}_{tag}"
    )
    cfg["ansatz"]["degree"] = int(degree)
    cfg["training"]["seed"] = int(seed)
    cfg["noise"]["training_rate"] = float(rate)
    cfg["noise"]["evaluation_rates"] = [float(rate)]
    cfg["paths"]["output_dir"] = str(
        output_root / f"degree_{int(degree):02d}" / tag
    )
    return cfg


def summarize_run(payload, result_file, degree, rate, seed):
    exact = min(
        payload["exact_density_evaluations"],
        key=lambda item: abs(
            float(item["depolarizing_rate"]) - float(rate)
        ),
    )
    finite = payload.get("finite_shot_evaluations", [])
    finite_record = (
        min(
            finite,
            key=lambda item: abs(
                float(item["depolarizing_rate"]) - float(rate)
            ),
        )
        if finite
        else None
    )
    selected_restart = int(payload["selected_restart"])
    selected = payload["restarts"][selected_restart]
    polish = selected.get("polish", {})
    return {
        "degree": int(degree),
        "depolarizing_rate": float(rate),
        "status": "ok",
        "seed": int(seed),
        "result_file": portable_path(result_file),
        "exact_ground_energy": float(
            payload["environment"]["model"]["exact_ground_energy"]
        ),
        "initialization": selected["initialization"],
        "phase_symmetry": payload["environment"]["ansatz"]["symmetry"],
        "selected_restart": selected_restart,
        "phi_su2": payload["best"]["phi_su2"],
        "trainable_phases": payload["best"]["trainable_phases"],
        "best_loss": float(payload["best"]["loss"]),
        "optimization_diagnostics": {
            "best_source": payload["best"].get("source", "adam"),
            "final_gradient_norm": payload["best"].get("gradient_norm"),
            "polish_passes": len(polish.get("passes", [])),
            "rescue_triggered": bool(polish.get("rescue_triggered", False)),
        },
        "success_penalty": float(payload["best"]["success_penalty"]),
        "exact_density_evaluation": exact,
        "pre_qsp_baseline": payload.get("pre_qsp_baseline"),
        "improvement_over_pre_qsp": payload.get(
            "improvement_over_pre_qsp"
        ),
        "ideal_phase_quality": payload["ideal_phase_quality"],
        "finite_shot_evaluation": finite_record,
        "resources": payload["resources"],
        "cross_validation": payload["cross_validation"],
    }


def initialize_summary(
    path,
    cfg,
    config_path,
    degrees,
    rates,
    overwrite,
    resume,
):
    if path.exists() and resume:
        summary = load_yaml(path)
        if summary.get("degrees") != degrees or summary.get("noise_rates") != rates:
            raise ValueError("Saved scan uses different degrees or noise rates.")
        if summary.get("config") != portable_metadata(cfg):
            raise ValueError(
                "Config changed since the saved scan. Use --overwrite to start over."
            )
        summary["status"] = "running"
        return summary
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists. Use --resume or --overwrite."
        )
    return {
        "schema_version": 3,
        "benchmark": cfg["experiment"],
        "status": "running",
        "started_at": timestamp(),
        "updated_at": timestamp(),
        "config_file": portable_path(config_path),
        "config": portable_metadata(cfg),
        "degrees": degrees,
        "noise_rates": rates,
        "training_protocol": {
            "initialization": cfg["initialization"]["method"],
            "phase_symmetry": cfg["ansatz"]["symmetry"],
            "same_initialization_across_noise_rates_at_fixed_degree": True,
            "seed_mode": "base_plus_degree",
            "loss": "normalized_measurement_model_energy",
            "success_probability_penalty_weight": 0.0,
            "shots_during_training": 0,
            "adam_iterations": int(cfg["training"]["iterations"]),
            "adaptive_lbfgs_polish": bool(
                cfg["training"].get("polish", {}).get("enabled", False)
            ),
        },
        "records": [],
    }


def validate_scan_config(cfg):
    validate_config(cfg)
    if float(cfg["loss"].get("success_penalty_weight", 0.0)) != 0.0:
        raise ValueError("Degree scan requires success_penalty_weight=0.")
    if cfg["loss"].get("type") != "energy":
        raise ValueError("Degree scan requires loss.type=energy.")
    degree_values(cfg["degree_scan"])
    _validate_rates(cfg["noise"]["sweep_rates"])


def degree_values(config):
    start = int(config["start"])
    stop = int(config["stop"])
    step = int(config["step"])
    return _validate_degrees(list(range(start, stop + 1, step)))


def _validate_degrees(values):
    if not values:
        raise ValueError("At least one degree is required.")
    degrees = sorted(set(int(value) for value in values))
    if any(degree < 2 or degree % 2 for degree in degrees):
        raise ValueError("All QETU degrees must be positive even integers.")
    return degrees


def _validate_rates(values):
    if not values:
        raise ValueError("At least one noise rate is required.")
    rates = sorted(set(float(value) for value in values))
    if any(
        not math.isfinite(rate) or rate < 0.0 or rate > 1.0
        for rate in rates
    ):
        raise ValueError("All depolarizing rates must lie in [0, 1].")
    return rates


def find_record(records, degree, rate, status=None):
    for record in records:
        if (
            int(record["degree"]) == int(degree)
            and abs(float(record["depolarizing_rate"]) - float(rate)) < 1.0e-15
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
        )
    ]
    records.append(new_record)


if __name__ == "__main__":
    main()
