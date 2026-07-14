from pathlib import Path
import copy

from io_utils import load_yaml, portable_metadata, write_yaml_atomic


ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "report_results"
DEGREES = list(range(2, 21, 2))

SOURCES = {
    "tfim_n2_density_training.yml": (
        ROOT
        / "results"
        / "dm_tfim_n2_g4_degree_scan_optimized"
        / "degree_scan_result.yml",
        "training",
    ),
    "tfim_n2_pre_qsp_comparison_shot20000.yml": (
        ROOT
        / "results"
        / "pre_qsp_vs_density_optimized_shot20000"
        / "comparison_shot20000.yml",
        "comparison",
    ),
    "tfim_n4_density_training.yml": (
        ROOT
        / "results"
        / "dm_tfim_n4_g4_degree_scan_optimized"
        / "degree_scan_result.yml",
        "training",
    ),
    "tfim_n4_pre_qsp_comparison_shot20000.yml": (
        ROOT
        / "results"
        / "pre_qsp_vs_density_n4_optimized_shot20000"
        / "comparison_shot20000.yml",
        "comparison",
    ),
}


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for output_name, (source_path, kind) in SOURCES.items():
        payload = load_yaml(source_path)
        if payload.get("status") != "complete":
            raise RuntimeError(f"Result is not complete: {source_path}")
        cleaned = clean_payload(payload, source_path, kind)
        output_path = REPORT_DIR / output_name
        write_yaml_atomic(output_path, cleaned)
        print(f"Wrote {output_path.relative_to(ROOT)}")


def clean_payload(payload, source_path, kind):
    cleaned = portable_metadata(copy.deepcopy(payload))
    cleaned.pop("config_file", None)
    cleaned["source_result"] = source_path.relative_to(ROOT).as_posix()
    cleaned["degrees"] = DEGREES
    cleaned["records"] = [
        record
        for record in cleaned["records"]
        if int(record["degree"]) in DEGREES
    ]
    cleaned.get("config", {}).pop("paths", None)
    drop_legacy_aliases(cleaned)
    if kind == "training":
        for record in cleaned["records"]:
            record.pop("result_file", None)
            record.pop("cross_validation", None)
    else:
        cleaned.get("runtime", {}).pop("python_executable", None)
    return cleaned


def drop_legacy_aliases(value):
    if isinstance(value, dict):
        value.pop("independent_phases", None)
        value.pop("num_independent_phases", None)
        for nested in value.values():
            drop_legacy_aliases(nested)
    elif isinstance(value, list):
        for nested in value:
            drop_legacy_aliases(nested)


if __name__ == "__main__":
    main()
