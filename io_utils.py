from datetime import datetime
from pathlib import Path
import os

import numpy as np
import torch
import yaml


YAML_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
YAML_DUMPER = getattr(yaml, "CSafeDumper", yaml.SafeDumper)
PROJECT_ROOT = Path(__file__).resolve().parent


def load_yaml(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.load(stream, Loader=YAML_LOADER)
    if not isinstance(payload, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return payload


def write_yaml_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        yaml.dump(
            to_builtin(payload),
            stream,
            Dumper=YAML_DUMPER,
            sort_keys=False,
            allow_unicode=False,
        )
    os.replace(temporary, path)


def save_torch_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def resolve_path(config_path, value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(config_path).resolve().parent / path
    return path.resolve()


def portable_path(path):
    """Return a repository-relative path for serialized metadata."""
    path = Path(path).expanduser().resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        try:
            return Path(os.path.relpath(path, PROJECT_ROOT)).as_posix()
        except ValueError:
            return path.name


def portable_metadata(value):
    """Remove machine-specific absolute paths from saved YAML metadata."""
    if isinstance(value, Path):
        return portable_path(value)
    if isinstance(value, str) and Path(value).expanduser().is_absolute():
        return portable_path(value)
    if isinstance(value, dict):
        return {
            str(key): portable_metadata(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [portable_metadata(item) for item in value]
    return value


def timestamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def to_builtin(value):
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.is_complex():
            return {
                "real": value.real.tolist(),
                "imag": value.imag.tolist(),
            }
        return value.tolist() if value.ndim else value.item()
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {"real": value.real.tolist(), "imag": value.imag.tolist()}
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    return value
