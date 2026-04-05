# TODO

## RL Training

- [ ] Unify surrogate interface so the same Gymnasium env can use RC, RF (TeacherRF), or PINN-lite.
  - **Design an interface** (python protocol or abstract class):
    - `reset(seed: Optional[int] = None, init_state: Optional[np.ndarray] = None) -> None`
    - `predict_next(state: np.ndarray, action: np.ndarray) -> float`  (or `step_dynamics(state, action) -> np.ndarray` if returning full next state)
  - **RFAdapter** (wraps TeacherRF):
    - Owns pandas history buffer and warmup length inference.
    - Builds features internally using `materialize_features_from_list()` based on the teacher's `feature_cols`.
    - Handles lags/rolling windows, state-to-base-feature mapping, and output-shape normalization (scalar temp).
    - Hides all RF-specific plumbing (`build_teacher_features_row`, history updates) from the Env.
  - **RCAdapter**:
    - Implements analytical/ODE-like thermal step given RC params.
    - No pandas history; just uses current state, action, and ambient/power inputs.
  - **PINNAdapter**:
    - Tensorizes inputs, applies any internal scaling, runs forward pass, returns float temperature.
    - Keeps model/device management inside the adapter.
  - **Env changes**:
    - Accept a `surrogate` object conforming to the interface.
    - Env remains responsible for reward, safety, curriculum, and logging.
    - Env calls only `surrogate.predict_next(state, action)`; never touches feature names, lags, or rolling.
  - **Curriculum & evaluation**:
    - Keep identical for all surrogates to ensure fair ablations.
    - Switch surrogate via config (e.g., `surrogate_type: rc|rf|pinn`, with per-type params).

- [ ] Refactor current `env_thermal_rf.py` to remove RF-specific feature materialization from the Env and move it into `RFAdapter`.
- [ ] Implement `RCAdapter` using calibrated RC parameters (expose config for R, C, dt; allow per-profile ambient/power modeling).
- [ ] Implement `PINNAdapter` (load lightweight PINN, handle device, inference, and optional normalization internally).
- [ ] Add config plumbing to select surrogate and pass its settings (e.g., `configs/rl_training.yaml`).
- [ ] Add unit tests for each adapter (`predict_next` determinism, shapes, warmup behavior for RF).
- [ ] Update training and eval scripts to work unchanged when surrogate switches (only config changes).
- [ ] Document the interface in `RL_IMPLEMENTATION_GUIDE.md` (architecture diagram + example config snippets).

## MPC Baseline

- [ ] Implement lightweight MPC controller for comparison (objective: track temp target, penalize fan effort & rate-of-change; hard constraints for safety).
- [ ] Integrate into evaluation harness for nominal and stress scenarios.

## Experiments & Evaluation

- [ ] Run 200k-step SAC training with Drive checkpoints; collect TensorBoard and eval logs.
- [ ] Comparative runs: Static Fan, Threshold, MPC, SAC (same scenarios, seeds, durations).
- [ ] Robustness suite: ambient shift, sensor noise, actuator lag, workload distribution shift.
- [ ] Ablations: with/without safety shield, reward weight sweeps, no curriculum vs curriculum, surrogate type sweep (RC/RF/PINN-lite).
- [ ] Export plots/tables for dissertation (throttle events, energy proxy, headroom, stability/oscillation, reward curves).
