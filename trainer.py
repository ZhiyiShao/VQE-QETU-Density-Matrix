import copy
import math
import time

import numpy as np
import torch

from qetu_density import metrics_to_python, wrap_phases


class DirectDensityMatrixTrainer:
    """Optimize QETU phases through exact PyTorch density-matrix autograd."""

    def __init__(self, model, training_cfg, loss_cfg, depolarizing_rate):
        self.model = model
        self.training_cfg = dict(training_cfg)
        self.loss_cfg = dict(loss_cfg)
        self.depolarizing_rate = float(depolarizing_rate)
        self.objective_evaluations = 0
        self._validate()

    def _validate(self):
        if str(self.training_cfg.get("optimizer", "adam")).lower() != "adam":
            raise ValueError("Density-matrix training currently supports optimizer=adam.")
        if int(self.training_cfg["iterations"]) <= 0:
            raise ValueError("training.iterations must be positive.")
        if float(self.training_cfg["learning_rate"]) <= 0.0:
            raise ValueError("training.learning_rate must be positive.")
        final_lr = float(
            self.training_cfg.get(
                "final_learning_rate",
                self.training_cfg["learning_rate"],
            )
        )
        if not 0.0 < final_lr <= float(self.training_cfg["learning_rate"]):
            raise ValueError("final_learning_rate must lie in (0, learning_rate].")
        if float(self.training_cfg.get("gradient_clip", 10.0)) <= 0.0:
            raise ValueError("training.gradient_clip must be positive.")
        for key in ("print_interval", "checkpoint_interval"):
            if int(self.training_cfg.get(key, 10)) <= 0:
                raise ValueError(f"training.{key} must be positive.")
        if int(self.training_cfg.get("history_phase_interval", 0)) < 0:
            raise ValueError("training.history_phase_interval cannot be negative.")
        early = dict(self.training_cfg.get("early_stopping", {}))
        if bool(early.get("enabled", False)):
            if int(early.get("patience", 30)) <= 0:
                raise ValueError("training.early_stopping.patience must be positive.")
            if float(early.get("min_delta", 1.0e-10)) < 0.0:
                raise ValueError("training.early_stopping.min_delta cannot be negative.")
        polish = dict(self.training_cfg.get("polish", {}))
        if bool(polish.get("enabled", False)):
            if str(polish.get("optimizer", "lbfgs")).lower() != "lbfgs":
                raise ValueError("training.polish.optimizer must be lbfgs.")
            if int(polish.get("max_iterations", 100)) <= 0:
                raise ValueError("training.polish.max_iterations must be positive.")
            if int(polish.get("max_evaluations", 125)) <= 0:
                raise ValueError("training.polish.max_evaluations must be positive.")
            if float(polish.get("learning_rate", 1.0)) <= 0.0:
                raise ValueError("training.polish.learning_rate must be positive.")
            if int(polish.get("history_size", 50)) <= 0:
                raise ValueError("training.polish.history_size must be positive.")
            if int(polish.get("max_passes", 1)) <= 0:
                raise ValueError("training.polish.max_passes must be positive.")
            if float(polish.get("rescue_gradient_norm", 1.0e-3)) <= 0.0:
                raise ValueError(
                    "training.polish.rescue_gradient_norm must be positive."
                )
            for key in ("rescue_max_iterations", "rescue_max_evaluations"):
                if key in polish and int(polish[key]) <= 0:
                    raise ValueError(f"training.polish.{key} must be positive.")
        loss_type = str(self.loss_cfg.get("type", "energy"))
        if loss_type not in ("energy", "infidelity", "hybrid"):
            raise ValueError("loss.type must be energy, infidelity, or hybrid.")
        energy_source = str(
            self.loss_cfg.get("energy_source", "measurement_model")
        )
        if energy_source not in ("measurement_model", "state"):
            raise ValueError("loss.energy_source must be measurement_model or state.")
        if float(self.loss_cfg.get("success_penalty_weight", 5.0)) < 0.0:
            raise ValueError("loss.success_penalty_weight cannot be negative.")
        if float(self.loss_cfg.get("infidelity_weight", 0.1)) < 0.0:
            raise ValueError("loss.infidelity_weight cannot be negative.")
        floor = float(self.loss_cfg.get("success_probability_floor", 0.05))
        if not 0.0 < floor <= 1.0:
            raise ValueError("success_probability_floor must lie in (0, 1].")

    def train(self, initial_phases, restart_index=0, checkpoint_callback=None):
        initial = torch.as_tensor(initial_phases, dtype=torch.float64).reshape(-1)
        if initial.numel() != self.model.num_trainable_phases:
            raise ValueError("Initial phase count does not match the QETU degree.")
        phases = torch.nn.Parameter(initial.clone().to(self.model.device))
        initial_learning_rate = float(self.training_cfg["learning_rate"])
        final_learning_rate = float(
            self.training_cfg.get("final_learning_rate", initial_learning_rate)
        )
        optimizer = torch.optim.Adam([phases], lr=initial_learning_rate)
        iterations = int(self.training_cfg["iterations"])
        gradient_clip = float(self.training_cfg.get("gradient_clip", 10.0))
        print_interval = int(self.training_cfg.get("print_interval", 10))
        checkpoint_interval = int(
            self.training_cfg.get("checkpoint_interval", print_interval)
        )
        early_cfg = dict(self.training_cfg.get("early_stopping", {}))
        early_enabled = bool(early_cfg.get("enabled", False))
        patience = int(early_cfg.get("patience", 30))
        min_delta = float(early_cfg.get("min_delta", 1.0e-10))

        history = []
        best = None
        stale_steps = 0
        started = time.perf_counter()

        for iteration in range(1, iterations + 1):
            learning_rate = _exponential_schedule(
                initial_learning_rate,
                final_learning_rate,
                iteration,
                iterations,
            )
            optimizer.param_groups[0]["lr"] = learning_rate
            loss, metrics, details = self.objective(phases)
            record = self._record(
                iteration,
                phases,
                loss,
                metrics,
                details,
                learning_rate,
            )
            if best is None or record["loss"] < best["loss"] - min_delta:
                best = copy.deepcopy(record)
                stale_steps = 0
            else:
                stale_steps += 1

            optimizer.zero_grad()
            loss.backward()
            if phases.grad is None or not torch.isfinite(phases.grad).all():
                raise RuntimeError("Encountered a non-finite phase gradient.")
            gradient_norm = torch.linalg.vector_norm(phases.grad)
            torch.nn.utils.clip_grad_norm_([phases], gradient_clip)
            record["gradient_norm"] = float(gradient_norm.detach().cpu())
            record["gradient_clipped"] = bool(
                float(gradient_norm.detach().cpu()) > gradient_clip
            )
            optimizer.step()
            with torch.no_grad():
                phases.copy_(wrap_phases(phases))
            history.append(self._history_record(record))

            if iteration == 1 or iteration % print_interval == 0 or iteration == iterations:
                print(_format_iteration(record))
            if checkpoint_callback is not None and (
                iteration % checkpoint_interval == 0 or iteration == iterations
            ):
                checkpoint_callback(
                    {
                        "restart_index": int(restart_index),
                        "iteration": int(iteration),
                        "history": history,
                        "best": best,
                        "current_trainable_phases": phases.detach().cpu(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    }
                )
            if early_enabled and stale_steps >= patience:
                break

        final_loss, final_metrics, final_details = self.objective(phases)
        final = self._record(
            len(history) + 1,
            phases,
            final_loss,
            final_metrics,
            final_details,
            float(optimizer.param_groups[0]["lr"]),
        )
        final["source"] = "after_last_update"
        if best is None or final["loss"] < best["loss"]:
            best = copy.deepcopy(final)

        polish, polished = self._polish(phases, best, len(history) + 2)
        if polished is not None and polished["loss"] < best["loss"]:
            best = copy.deepcopy(polished)
        elapsed = time.perf_counter() - started
        return {
            "restart_index": int(restart_index),
            "initial_trainable_phases": initial.tolist(),
            "history": history,
            "polish": polish,
            "best": best,
            "final": final,
            "iterations_completed": len(history),
            "stopped_early": bool(len(history) < iterations),
            "objective_evaluations": int(self.objective_evaluations),
            "training_shots": 0,
            "wall_time_seconds": float(elapsed),
        }

    def _polish(self, phases, best, iteration):
        config = dict(self.training_cfg.get("polish", {}))
        if not bool(config.get("enabled", False)):
            return {"enabled": False}, None

        max_passes = int(config.get("max_passes", 1))
        rescue_gradient_norm = float(
            config.get("rescue_gradient_norm", 1.0e-3)
        )
        start = copy.deepcopy(best)
        best_polished = None
        passes = []
        for pass_index in range(max_passes):
            candidate, report = self._lbfgs_pass(
                phases,
                start,
                iteration + pass_index,
                config,
                pass_index,
            )
            passes.append(report)
            if best_polished is None or candidate["loss"] < best_polished["loss"]:
                best_polished = copy.deepcopy(candidate)
            if candidate["loss"] < start["loss"]:
                start = copy.deepcopy(candidate)
            if candidate["gradient_norm"] <= rescue_gradient_norm:
                break

        evaluations = sum(item["objective_evaluations"] for item in passes)
        return {
            "enabled": True,
            "optimizer": "lbfgs",
            "max_passes": max_passes,
            "rescue_gradient_norm": rescue_gradient_norm,
            "rescue_triggered": len(passes) > 1,
            "objective_evaluations": int(evaluations),
            "passes": passes,
            "start": copy.deepcopy(best),
            "final": copy.deepcopy(best_polished),
            "improved": bool(best_polished["loss"] < best["loss"]),
        }, best_polished

    def _lbfgs_pass(
        self,
        phases,
        start,
        iteration,
        config,
        pass_index,
    ):
        with torch.no_grad():
            phases.copy_(
                torch.as_tensor(
                    start["trainable_phases"],
                    dtype=torch.float64,
                    device=self.model.device,
                )
            )
        is_rescue = pass_index > 0
        max_iterations_key = (
            "rescue_max_iterations" if is_rescue else "max_iterations"
        )
        max_evaluations_key = (
            "rescue_max_evaluations" if is_rescue else "max_evaluations"
        )
        max_iterations = int(
            config.get(max_iterations_key, config.get("max_iterations", 100))
        )
        max_evaluations = int(
            config.get(
                max_evaluations_key,
                config.get("max_evaluations", max_iterations * 5 // 4),
            )
        )
        line_search = config.get("line_search", "strong_wolfe")
        if line_search in ("none", "None", ""):
            line_search = None
        optimizer = torch.optim.LBFGS(
            [phases],
            lr=float(config.get("learning_rate", 1.0)),
            max_iter=max_iterations,
            max_eval=max_evaluations,
            tolerance_grad=float(config.get("tolerance_grad", 1.0e-10)),
            tolerance_change=float(config.get("tolerance_change", 1.0e-12)),
            history_size=int(config.get("history_size", 50)),
            line_search_fn=line_search,
        )
        evaluations_before = self.objective_evaluations

        def closure():
            optimizer.zero_grad()
            loss, _, _ = self.objective(phases)
            loss.backward()
            if phases.grad is None or not torch.isfinite(phases.grad).all():
                raise RuntimeError("Encountered a non-finite L-BFGS gradient.")
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            phases.copy_(wrap_phases(phases))
        optimizer.zero_grad()
        loss, metrics, details = self.objective(phases)
        loss.backward()
        if phases.grad is None or not torch.isfinite(phases.grad).all():
            raise RuntimeError("Encountered a non-finite polished gradient.")
        gradient_norm = float(torch.linalg.vector_norm(phases.grad).detach().cpu())
        polished = self._record(
            iteration,
            phases,
            loss,
            metrics,
            details,
            float(config.get("learning_rate", 1.0)),
        )
        polished["source"] = (
            "lbfgs_polish" if pass_index == 0 else "lbfgs_rescue"
        )
        polished["gradient_norm"] = gradient_norm
        evaluations = self.objective_evaluations - evaluations_before
        print(
            f"L-BFGS pass {pass_index + 1} | "
            f"E={polished['metrics']['energy_measurement_model']:+.10f} | "
            f"scaled_error={polished['normalized_energy']:.3e} | "
            f"grad={gradient_norm:.3e} | evaluations={evaluations}"
        )
        return polished, {
            "pass_index": int(pass_index),
            "kind": "rescue" if is_rescue else "primary",
            "max_iterations": max_iterations,
            "max_evaluations": max_evaluations,
            "objective_evaluations": int(evaluations),
            "start_loss": float(start["loss"]),
            "final_loss": float(polished["loss"]),
            "gradient_norm": gradient_norm,
            "final": copy.deepcopy(polished),
            "improved": bool(polished["loss"] < start["loss"]),
        }

    def objective(self, trainable_phases):
        metrics = self.model.simulate(
            trainable_phases,
            depolarizing_rate=self.depolarizing_rate,
            measurement_basis_noise=bool(
                self.loss_cfg.get("include_measurement_basis_noise", True)
            ),
        )
        self.objective_evaluations += 1
        energy_key = (
            "energy_measurement_model"
            if str(self.loss_cfg.get("energy_source", "measurement_model"))
            == "measurement_model"
            else "energy_state"
        )
        energy = metrics[energy_key]
        width = self.model.scaling.energy_high - self.model.scaling.energy_low
        normalized_energy = (
            energy - self.model.scaling.energy_low
        ) / width
        infidelity = 1.0 - metrics["fidelity_ground_space"]
        loss_type = str(self.loss_cfg.get("type", "energy"))
        if loss_type == "energy":
            base = normalized_energy
        elif loss_type == "infidelity":
            base = infidelity
        else:
            base = normalized_energy + float(
                self.loss_cfg.get("infidelity_weight", 0.1)
            ) * infidelity

        floor = float(self.loss_cfg.get("success_probability_floor", 0.05))
        deficit = torch.relu(
            torch.as_tensor(floor, dtype=torch.float64, device=self.model.device)
            - metrics["success_probability_ancilla0"]
        ) / floor
        success_weight = float(
            self.loss_cfg.get("success_penalty_weight", 5.0)
        )
        success_penalty = (
            torch.zeros_like(base)
            if success_weight == 0.0
            else success_weight * deficit**2
        )
        total = base + success_penalty
        details = {
            "loss_type": loss_type,
            "energy_source": energy_key,
            "base_loss": base,
            "normalized_energy": normalized_energy,
            "infidelity": infidelity,
            "success_penalty": success_penalty,
        }
        return total, metrics, details

    def _record(
        self,
        iteration,
        phases,
        loss,
        metrics,
        details,
        learning_rate,
    ):
        metrics_python = metrics_to_python(metrics)
        trainable = phases.detach().cpu().numpy()
        full = metrics["phi_su2"].detach().cpu().numpy()
        return {
            "iteration": int(iteration),
            "loss": float(loss.detach().cpu()),
            "base_loss": float(details["base_loss"].detach().cpu()),
            "success_penalty": float(
                details["success_penalty"].detach().cpu()
            ),
            "normalized_energy": float(
                details["normalized_energy"].detach().cpu()
            ),
            "infidelity": float(details["infidelity"].detach().cpu()),
            "learning_rate": float(learning_rate),
            "trainable_phases": trainable.tolist(),
            "phi_su2": full.tolist(),
            "metrics": metrics_python,
        }

    def _history_record(self, record):
        """Keep scalar diagnostics in YAML; phases remain in best/final/checkpoints."""
        phase_keys = ("trainable_phases", "phi_su2")
        compact = {
            key: copy.deepcopy(value)
            for key, value in record.items()
            if key not in phase_keys
        }
        interval = int(self.training_cfg.get("history_phase_interval", 0))
        if interval and int(record["iteration"]) % interval == 0:
            for key in phase_keys:
                compact[key] = copy.deepcopy(record[key])
        return compact


def initialize_phases(model, initialization_cfg, seed, pre_qsp_record=None):
    config = dict(initialization_cfg)
    method = str(config.get("method", "random_uniform"))
    count = model.num_trainable_phases
    if method == "random_uniform":
        bound = float(config.get("bound", 0.25))
        if not math.isfinite(bound) or bound <= 0.0:
            raise ValueError("initialization.bound must be finite and positive.")
        rng = np.random.default_rng(int(seed))
        phases = rng.uniform(-bound, bound, count)
    elif method == "zeros":
        phases = np.zeros(count, dtype=float)
    elif method == "pi_quarter_endpoints":
        phases = np.zeros(count, dtype=float)
        phases[0] = math.pi / 4.0
        if model.phase_symmetry == "full":
            phases[-1] = math.pi / 4.0
    elif method == "pre_qsp":
        if pre_qsp_record is None:
            raise ValueError("pre_qsp initialization requires a phase record.")
        full = np.asarray(
            pre_qsp_record["phases"]["phi_su2_for_qetu_api"],
            dtype=float,
        )
        if full.size != model.degree + 1 or not np.all(np.isfinite(full)):
            raise ValueError("Pre-QSP phase count does not match the degree.")
        phases = (
            full[:count]
            if model.phase_symmetry == "palindrome"
            else full
        )
    else:
        raise ValueError(
            "initialization.method must be random_uniform, zeros, "
            "pi_quarter_endpoints, or pre_qsp."
        )
    return np.asarray(phases, dtype=float), {
        "method": method,
        "seed": int(seed),
        "phase_symmetry": model.phase_symmetry,
        "trainable_phases": np.asarray(phases, dtype=float).tolist(),
    }


def _format_iteration(record):
    metrics = record["metrics"]
    exact = record["normalized_energy"]
    return (
        f"iter {record['iteration']:4d} | "
        f"E={metrics['energy_measurement_model']:+.10f} | "
        f"state_E={metrics['energy_state']:+.10f} | "
        f"scaled_error={exact:.3e} | "
        f"F={metrics['fidelity_ground_space']:.7f} | "
        f"p0={metrics['success_probability_ancilla0']:.4f} | "
        f"lr={record['learning_rate']:.2e}"
    )


def _exponential_schedule(initial, final, iteration, iterations):
    if iterations <= 1:
        return float(final)
    progress = (int(iteration) - 1) / (int(iterations) - 1)
    return float(initial * (final / initial) ** progress)
