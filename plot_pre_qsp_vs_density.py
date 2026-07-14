from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from io_utils import load_yaml


METHODS = ("pre_qsp", "density_matrix_training")
METHOD_TITLES = {
    "pre_qsp": "pre-QSP phases",
    "density_matrix_training": "Density-matrix-trained phases",
}
METHOD_LINESTYLES = {
    "pre_qsp": "--",
    "density_matrix_training": "-",
}
RATE_STYLES = {
    1.0e-5: {"color": "#8c564b", "marker": "|", "label": r"$r_{dplz}=10^{-5}$"},
    1.0e-4: {"color": "#2ca02c", "marker": "+", "label": r"$r_{dplz}=10^{-4}$"},
    1.0e-3: {"color": "#ff7f0e", "marker": "v", "label": r"$r_{dplz}=10^{-3}$"},
    1.0e-2: {"color": "#1f77b4", "marker": "o", "label": r"$r_{dplz}=10^{-2}$"},
}


def main():
    parser = argparse.ArgumentParser(
        description="Plot the report-ready pre-QSP versus trained-phase comparison."
    )
    parser.add_argument("--result", required=True)
    parser.add_argument("--exact-output")
    parser.add_argument("--shot-output")
    args = parser.parse_args()

    result_path = Path(args.result).expanduser().resolve()
    payload = load_yaml(result_path)
    validate_payload(payload)
    records = [item for item in payload["records"] if item.get("status") == "ok"]
    figure_dir = Path(__file__).resolve().parent / "figures"
    tag = f"tfim_n{int(payload['model']['n_qubits'])}"
    exact_path = output_path(
        args.exact_output,
        figure_dir / f"{tag}_pre_qsp_vs_density_exact_density_noisy_only.png",
    )
    shot_path = output_path(
        args.shot_output,
        figure_dir / f"{tag}_pre_qsp_vs_density_finite_shot_noisy_only.png",
    )

    plot_comparison(payload, records, exact_path, finite_shots=False)
    plot_comparison(payload, records, shot_path, finite_shots=True)
    print(f"Exact-density comparison written to: {exact_path}")
    print(f"Finite-shot comparison written to: {shot_path}")


def output_path(value, default):
    return Path(value).expanduser().resolve() if value else default


def plot_comparison(payload, records, output, finite_shots):
    exact_ground = float(payload["model"]["exact_ground_energy"])
    rates = noisy_rates(payload["noise_rates"])
    figure, (error_axis, gain_axis) = plt.subplots(
        2,
        1,
        figsize=(9.2, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": (2.2, 1.0)},
    )

    for rate in rates:
        style = rate_style(rate)
        errors_by_method = {}
        for method in METHODS:
            degrees, errors, standard_deviations = result_series(
                records,
                method,
                rate,
                exact_ground,
                finite_shots,
            )
            errors_by_method[method] = errors
            error_axis.errorbar(
                degrees,
                errors,
                yerr=positive_errorbars(errors, standard_deviations),
                color=style["color"],
                marker=style["marker"],
                linestyle=METHOD_LINESTYLES[method],
                linewidth=1.5,
                markersize=4.5,
                capsize=2.0 if finite_shots else 0.0,
            )

        pre_error = errors_by_method["pre_qsp"]
        trained_error = errors_by_method["density_matrix_training"]
        reduction = 100.0 * (pre_error - trained_error) / pre_error
        gain_axis.plot(
            degrees,
            reduction,
            color=style["color"],
            marker=style["marker"],
            linewidth=1.4,
            markersize=4.5,
        )

    error_axis.set_yscale("log")
    error_axis.set_ylabel(r"$|E_{grd}-E_{grd}^{exact}|$")
    error_axis.set_title(figure_title(payload, finite_shots), fontsize=13)
    error_axis.grid(alpha=0.25, which="both")
    noise_legend = error_axis.legend(
        handles=noise_legend_handles(rates),
        loc="lower right",
        frameon=True,
    )
    error_axis.add_artist(noise_legend)
    error_axis.legend(
        handles=method_legend_handles(),
        loc="upper right",
        frameon=True,
    )

    gain_axis.axhline(0.0, color="#555555", linestyle=":", linewidth=1.0)
    gain_axis.set_ylabel("Error reduction (%)")
    gain_axis.set_xlabel(r"QETU degree $d$")
    gain_axis.set_xticks(payload["degrees"])
    gain_axis.grid(alpha=0.25)
    figure.subplots_adjust(
        left=0.11,
        right=0.98,
        top=0.90,
        bottom=0.10,
        hspace=0.08,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def result_series(records, method, rate, exact_ground, finite_shots):
    selected = sorted(
        (
            item
            for item in records
            if item["method"] == method
            and np.isclose(
                float(item["depolarizing_rate"]),
                float(rate),
                rtol=0.0,
                atol=1.0e-15,
            )
        ),
        key=lambda item: int(item["degree"]),
    )
    degrees = np.asarray([item["degree"] for item in selected], dtype=int)
    if finite_shots:
        evaluations = [item["finite_shot_evaluation"] for item in selected]
        energies = np.asarray(
            [float(item["energy_mean"]) for item in evaluations]
        )
        standard_deviations = np.asarray(
            [float(item["energy_standard_deviation"]) for item in evaluations]
        )
    else:
        evaluations = [item["exact_density_evaluation"] for item in selected]
        energies = np.asarray(
            [float(item["energy_measurement_model"]) for item in evaluations]
        )
        standard_deviations = None
    errors = np.maximum(np.abs(energies - exact_ground), 1.0e-12)
    return degrees, errors, standard_deviations


def positive_errorbars(values, standard_deviations):
    if standard_deviations is None:
        return None
    # On a log axis, suppress a lower error segment that would cross zero.
    lower = np.where(standard_deviations < values, standard_deviations, 0.0)
    return np.vstack((lower, standard_deviations))


def figure_title(payload, finite_shots):
    n_qubits = int(payload["model"]["n_qubits"])
    if not finite_shots:
        subtitle = "Exact density-matrix expectation (no shot noise)"
    else:
        evaluation = payload["config"]["evaluation"]
        subtitle = (
            f"{int(evaluation['shots_per_basis']):,} shots/basis, "
            f"{int(evaluation['repetitions'])} repetitions"
        )
    return f"{n_qubits}-qubit TFIM QETU noisy energy error\n{subtitle}"


def noisy_rates(rates):
    return sorted(
        (
            float(rate)
            for rate in rates
            if not np.isclose(float(rate), 0.0, rtol=0.0, atol=1.0e-15)
        ),
        reverse=True,
    )


def rate_style(rate):
    for reference, style in RATE_STYLES.items():
        if np.isclose(float(rate), reference, rtol=0.0, atol=1.0e-15):
            return style
    raise ValueError(f"No plot style is defined for r_dplz={rate}.")


def noise_legend_handles(rates):
    return [
        Line2D(
            [],
            [],
            color=rate_style(rate)["color"],
            marker=rate_style(rate)["marker"],
            linestyle="-",
            linewidth=1.0,
            markersize=5.0,
            label=rate_style(rate)["label"],
        )
        for rate in rates
    ]


def method_legend_handles():
    return [
        Line2D(
            [],
            [],
            color="#222222",
            linestyle=METHOD_LINESTYLES[method],
            linewidth=1.6,
            label=METHOD_TITLES[method],
        )
        for method in METHODS
    ]


def validate_payload(payload):
    if payload.get("status") != "complete":
        raise RuntimeError(
            f"Comparison is not complete (status={payload.get('status')!r}). "
            "Resume evaluate_pre_qsp_vs_density.py before plotting."
        )
    records = [item for item in payload["records"] if item.get("status") == "ok"]
    expected = (
        len(payload["degrees"])
        * len(payload["noise_rates"])
        * len(payload["methods"])
    )
    if len(records) != expected:
        raise ValueError(f"Comparison is incomplete: expected {expected}, got {len(records)}.")
    if tuple(payload["methods"]) != METHODS:
        raise ValueError(f"Expected methods {METHODS}, got {payload['methods']}.")
    for rate in noisy_rates(payload["noise_rates"]):
        rate_style(rate)


if __name__ == "__main__":
    main()
