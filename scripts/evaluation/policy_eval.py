"""
Evaluation harness for MPC and RL policies across multiple surrogates.

Implements:
- Scenario generation (nominal + stress)
- Closed-loop rollout with MPC and RL (SAC) controllers
- Metrics: temperature tracking, violations, fan effort, smoothness, reward
- Artifacts: per-episode CSV, per-episode plots, per-scenario and overall CSV summaries

Examples:
  # Evaluate all four controllers (MPC-RC, MPC-RCNN, RL-RC, RL-RCNN)
  python scripts/evaluation/policy_eval.py \
    --eval mpc_rc mpc_rcnn rl_rc rl_rcnn \
    --rl-rc-model runs/rl/sac_rc_baseline/checkpoints/sac_170000_steps.zip \
    --rl-rc-vecnorm runs/rl/sac_rc_baseline/vecnormalize.pkl \
    --rl-rcnn-model runs/rl/sac_rcnn_hybrid/checkpoints/sac_195000_steps.zip \
    --rl-rcnn-vecnorm runs/rl/sac_rcnn_hybrid/vecnormalize.pkl \
    --rcnn-bundle models/rc_nn_hybrid.pkl \
    --output-dir results/policy_eval \
    --episodes 10
"""

import argparse
from pathlib import Path
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from src.control.mpc_controller import MPCController
from src.rl.environments.thermal_unified import ThermalControlEnv
from src.rl.surrogates.factory import create_surrogate


# -------- Scenario generation --------

def scenario_schedule(name: str, T: int):
    """Return ambient and power schedules for a scenario over T steps.
    Outputs arrays ambient[T], power[T]. Baselines: ambient=25, power=200.
    """
    amb = np.full(T, 25.0, dtype=float)
    poww = np.full(T, 200.0, dtype=float)

    if name == "baseline":
        pass
    elif name == "low_workload":
        poww[:] = 120.0
    elif name == "high_workload":
        poww[:] = 300.0
    elif name == "variable_workload":
        for t in range(T):
            poww[t] = 200.0 + 60.0 * math.sin(2 * math.pi * t / 60.0)
    elif name == "warm_ambient":
        amb[:] = 30.0
    elif name == "thermal_high_start":
        # start hot: handled via env reset override; keep workload moderate
        pass
    elif name == "ambient_extreme":
        amb[:] = 33.0
    elif name == "workload_spike":
        poww[:] = 180.0
        spike_idx = np.arange(50, min(70, T))
        poww[spike_idx] = 320.0
    elif name == "workload_oscillation":
        for t in range(T):
            poww[t] = 220.0 + 90.0 * math.sin(2 * math.pi * t / 20.0)
    elif name == "combined_extreme":
        amb[:] = 32.0
        for t in range(T):
            poww[t] = 240.0 + 70.0 * math.sin(2 * math.pi * t / 30.0)
    elif name == "recovery":
        # start hot then cool ambient and workload gradually
        for t in range(T):
            amb[t] = 31.0 - 0.02 * t
            poww[t] = 260.0 - 0.5 * t
    elif name == "sustained_limit":
        amb[:] = 31.0
        poww[:] = 280.0
    else:
        raise ValueError(f"Unknown scenario: {name}")
    return amb, poww


NOMINAL_SCENARIOS = [
    "baseline",
    "low_workload",
    "high_workload",
    "variable_workload",
    "warm_ambient",
]

STRESS_SCENARIOS = [
    "thermal_high_start",
    "ambient_extreme",
    "workload_spike",
    "workload_oscillation",
    "combined_extreme",
    "recovery",
    "sustained_limit",
]


# -------- Evaluation helpers --------

def make_env(surrogate_type: str, surrogate_cfg: dict, env_cfg: dict) -> ThermalControlEnv:
    surrogate = create_surrogate(surrogate_type, surrogate_cfg)
    env = ThermalControlEnv(surrogate=surrogate, config=env_cfg)
    return env


def rollout_episode(policy_env, base_env: ThermalControlEnv, controller, controller_type: str, T: int, amb_sched, pow_sched, seed: int, scenario: str,
                    temp_warning: float, temp_critical: float, action_delay: int = 0):
    # Reset base env with seed for reproducibility
    _obs, _info = base_env.reset(seed=seed)
    # Reset policy env (handles VecNormalize/DummyVecEnv)
    if controller_type == "rl":
        obs = policy_env.reset()
        # VecEnv returns batched obs; take first env
        if isinstance(obs, np.ndarray) and obs.ndim > 1:
            obs_current = obs[0]
        else:
            obs_current = obs
    else:
        obs = _obs
        obs_current = obs
    # Stress: thermal_high_start -> lift initial temp
    if scenario == "thermal_high_start":
        base_env.state[0] = max(base_env.state[0], base_env.temp_warning + 2.0)

    traj = {
        "t": [], "temp": [], "ambient": [], "power": [], "fan": [], "reward": [],
        "thermal_violation": [], "critical_violation": []
    }
    # Action delay buffer: start with sensible default (e.g., last action = current fan)
    prev_action = np.array([base_env.state[3]], dtype=np.float32)
    for t in range(T):
        # Inject scenario ambient/power for this step
        base_env.state[1] = float(amb_sched[t])
        base_env.state[2] = float(pow_sched[t])

        if controller_type == "mpc":
            action = controller.act(base_env.state)  # implement .act on MPCController via a small shim below
            # Apply delay
            env_action = prev_action if action_delay > 0 else action
            obs_next, reward, terminated, truncated, info = base_env.step(env_action)
            prev_action = action
            obs_current = obs_next
        else:
            a, _ = controller.predict(obs_current, deterministic=True)
            # VecEnv expects actions batched as (n_envs, action_dim); ensure shape (1,1)
            if np.isscalar(a):
                action = np.array([[float(a)]], dtype=np.float32)
            else:
                arr = np.asarray(a, dtype=np.float32).reshape(-1)
                if arr.size == 0:
                    action = np.array([[0.0]], dtype=np.float32)
                else:
                    action = arr[:1].reshape(1, -1)
            # Apply delay
            env_action = prev_action.reshape(1, -1) if action_delay > 0 else action
            obs_next, rewards, dones, infos = policy_env.step(env_action)
            # Update delay buffer with intended action (not applied this step when delayed)
            prev_action = action.reshape(-1)
            # VecEnv outputs batched arrays
            if isinstance(obs_next, np.ndarray) and obs_next.ndim > 1:
                obs_current = obs_next[0]
                reward = float(rewards[0])
                done_flag = bool(dones[0])
                info = infos[0] if isinstance(infos, (list, tuple)) and len(infos) else {}
            else:
                obs_current = obs_next
                reward = float(rewards)
                done_flag = bool(dones)
                info = infos if isinstance(infos, dict) else {}

            # Mirror termination logic using base env step threshold when using VecEnv
            terminated = done_flag
            truncated = False

        traj["t"].append(t)
        traj["temp"].append(float(base_env.state[0]))
        traj["ambient"].append(float(base_env.state[1]))
        traj["power"].append(float(base_env.state[2]))
        traj["fan"].append(float(base_env.state[3]))
        traj["reward"].append(float(reward))
        traj["thermal_violation"].append(bool(info.get("thermal_violation", False)))
        traj["critical_violation"].append(bool(info.get("critical_violation", False)))

        if terminated or truncated:
            break

    df = pd.DataFrame(traj)
    # Metrics
    warn_series = df["thermal_violation"].astype(bool)
    crit_series = df["critical_violation"].astype(bool)
    # Duration (steps above threshold)
    warn_duration = int(warn_series.sum())
    crit_duration = int(crit_series.sum())
    # Entries (count crossings from non-violation to violation)
    warn_entries = int(((~warn_series.shift(fill_value=False)) & warn_series).sum())
    crit_entries = int(((~crit_series.shift(fill_value=False)) & crit_series).sum())
    # Do not count the initial state as an 'entry' if we start hot
    if len(df) > 0 and df.loc[0, "temp"] > temp_warning:
        warn_entries = max(0, warn_entries - 1)
    if len(df) > 0 and df.loc[0, "temp"] > temp_critical:
        crit_entries = max(0, crit_entries - 1)
    # Time-to-safe-band: first t from which we remain below warning for rest of episode
    time_to_safe = None
    if len(df) > 0:
        # Find indices where temp <= warning
        below = ~(df["temp"] > temp_warning)
        # From the end backwards, build suffix-AND to mark positions after which we never go above warning
        suffix_all_below = np.flip(np.cumprod(np.flip(below.values).astype(int))).astype(bool)
        candidates = np.where(suffix_all_below & below.values)[0]
        if candidates.size > 0:
            time_to_safe = int(candidates[0])
    metrics = {
        "episode_length": len(df),
        "mean_temp": float(df["temp"].mean()),
        "max_temp": float(df["temp"].max()),
        "min_temp": float(df["temp"].min()),
        "violations_warning": warn_duration,
        "violations_critical": crit_duration,
        "violations_warning_entries": warn_entries,
        "violations_critical_entries": crit_entries,
        "time_to_safe_band": (time_to_safe if time_to_safe is not None else -1),
        "mean_fan": float(df["fan"].mean()),
        "sum_fan": float(df["fan"].sum()),
        "fan_smoothness": float(np.abs(np.diff(df["fan"])).mean()) if len(df) > 1 else 0.0,
        "cumulative_reward": float(df["reward"].sum()),
    }
    return df, metrics


def plot_episode(df: pd.DataFrame, out_dir: Path, scenario: str, tag: str, temp_warning: float, temp_critical: float, episode: int = 0):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Temp plot
    plt.figure(figsize=(9,4))
    plt.plot(df["t"], df["temp"], label="temp")
    plt.plot(df["t"], df["ambient"], label="ambient", alpha=0.6)
    plt.axhline(temp_warning, color="orange", linestyle="--", label="warning")
    plt.axhline(temp_critical, color="red", linestyle=":", label="critical")
    plt.xlabel("t")
    plt.ylabel("Temp (C)")
    plt.title(f"{tag} | {scenario} | ep{episode}: Temperature")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(out_dir / f"{tag}_{scenario}_ep{episode}_temp.png", dpi=150); plt.close()

    # Fan plot
    plt.figure(figsize=(9,3.5))
    plt.plot(df["t"], df["fan"], color="tab:green")
    plt.xlabel("t"); plt.ylabel("Fan %")
    plt.title(f"{tag} | {scenario} | ep{episode}: Fan")
    plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(out_dir / f"{tag}_{scenario}_ep{episode}_fan.png", dpi=150); plt.close()


def summarize_and_save(all_metrics: list, out_dir: Path, tag: str):
    df = pd.DataFrame(all_metrics)
    df.to_csv(out_dir / f"{tag}_episodes.csv", index=False)
    # Per-scenario summary
    agg = df.groupby("scenario").agg({
        "episode_length": "mean",
        "mean_temp": "mean",
        "max_temp": "mean",
        "violations_warning": "mean",
        "violations_critical": "mean",
        "mean_fan": "mean",
        "sum_fan": "mean",
        "fan_smoothness": "mean",
        "cumulative_reward": "mean",
    }).reset_index()
    agg.to_csv(out_dir / f"{tag}_per_scenario.csv", index=False)
    return df, agg


def mpc_act_shim(mpc: MPCController):
    # Add a light shim so we can call mpc.act(state)
    def _act(state: np.ndarray):
        val = mpc.compute_action(state)
        # Extract first numeric scalar from potentially nested structure
        def first_scalar(x):
            if x is None:
                return 0.0
            if np.isscalar(x):
                return float(x)
            if isinstance(x, (list, tuple)) and len(x) > 0:
                return first_scalar(x[0])
            try:
                arr = np.array(x)
                if arr.size == 0:
                    return 0.0
                return float(arr.flatten()[0])
            except Exception:
                return 0.0
        a0 = first_scalar(val)
        return np.array([a0], dtype=np.float32)
    setattr(mpc, "act", _act)
    return mpc


def plot_combined_multi_episode(output_dir: Path, episodes_dir: Path, controllers: list, scenarios: list, temp_warning: float, temp_critical: float):
    """
    Plot all episodes for each controller-scenario combination on one plot (with mean ± std).
    Saves to output_dir/combined_episodes/
    """
    combined_dir = output_dir / "combined_episodes"
    combined_dir.mkdir(exist_ok=True)
    
    for tag in controllers:
        for scen in scenarios:
            # Load all episode CSVs for this controller-scenario
            episode_dfs = []
            for ep_file in sorted(episodes_dir.glob(f"{tag}_{scen}_ep*.csv")):
                episode_dfs.append(pd.read_csv(ep_file))
            
            if not episode_dfs:
                continue
            
            # Temperature plot with all episodes + mean
            plt.figure(figsize=(10,5))
            temps = []
            for i, df in enumerate(episode_dfs):
                plt.plot(df["t"], df["temp"], alpha=0.3, color="tab:blue")
                temps.append(df["temp"].values)
            
            # Mean and std
            temps_arr = np.array(temps)
            mean_temp = temps_arr.mean(axis=0)
            std_temp = temps_arr.std(axis=0)
            t = episode_dfs[0]["t"].values
            plt.plot(t, mean_temp, color="tab:blue", linewidth=2, label=f"mean (n={len(episode_dfs)})")
            plt.fill_between(t, mean_temp - std_temp, mean_temp + std_temp, alpha=0.2, color="tab:blue")
            plt.axhline(temp_warning, color="orange", linestyle="--", label="warning", linewidth=1.5)
            plt.axhline(temp_critical, color="red", linestyle=":", label="critical", linewidth=1.5)
            plt.xlabel("Time (s)")
            plt.ylabel("Temperature (°C)")
            plt.title(f"{tag} | {scen}: Temperature (all episodes)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(combined_dir / f"{tag}_{scen}_temp_all_episodes.png", dpi=150)
            plt.close()
            
            # Fan plot with all episodes + mean
            plt.figure(figsize=(10,4))
            fans = []
            for i, df in enumerate(episode_dfs):
                plt.plot(df["t"], df["fan"], alpha=0.3, color="tab:green")
                fans.append(df["fan"].values)
            
            fans_arr = np.array(fans)
            mean_fan = fans_arr.mean(axis=0)
            std_fan = fans_arr.std(axis=0)
            plt.plot(t, mean_fan, color="tab:green", linewidth=2, label=f"mean (n={len(episode_dfs)})")
            plt.fill_between(t, mean_fan - std_fan, mean_fan + std_fan, alpha=0.2, color="tab:green")
            plt.xlabel("Time (s)")
            plt.ylabel("Fan Speed (%)")
            plt.title(f"{tag} | {scen}: Fan Speed (all episodes)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(combined_dir / f"{tag}_{scen}_fan_all_episodes.png", dpi=150)
            plt.close()


def plot_controller_comparison_per_scenario(output_dir: Path, episodes_dir: Path, controllers: list, scenarios: list, temp_warning: float, temp_critical: float):
    """
    For each scenario, plot all controllers on one plot (mean trajectory).
    Saves to output_dir/controller_comparison/
    """
    comp_dir = output_dir / "controller_comparison"
    comp_dir.mkdir(exist_ok=True)
    
    colors = {"mpc_rc": "tab:blue", "mpc_rcnn": "tab:cyan", "rl_rc": "tab:orange", "rl_rcnn": "tab:red"}
    
    for scen in scenarios:
        # Temperature comparison
        plt.figure(figsize=(10,5))
        for tag in controllers:
            episode_dfs = [pd.read_csv(f) for f in sorted(episodes_dir.glob(f"{tag}_{scen}_ep*.csv"))]
            if not episode_dfs:
                continue
            temps = np.array([df["temp"].values for df in episode_dfs])
            mean_temp = temps.mean(axis=0)
            t = episode_dfs[0]["t"].values
            plt.plot(t, mean_temp, label=tag, color=colors.get(tag, None), linewidth=2)
        
        plt.axhline(temp_warning, color="orange", linestyle="--", label="warning", linewidth=1, alpha=0.7)
        plt.axhline(temp_critical, color="red", linestyle=":", label="critical", linewidth=1, alpha=0.7)
        plt.xlabel("Time (s)")
        plt.ylabel("Temperature (°C)")
        plt.title(f"{scen}: Controller Comparison (Temperature)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(comp_dir / f"{scen}_temp_comparison.png", dpi=150)
        plt.close()
        
        # Fan comparison
        plt.figure(figsize=(10,4))
        for tag in controllers:
            episode_dfs = [pd.read_csv(f) for f in sorted(episodes_dir.glob(f"{tag}_{scen}_ep*.csv"))]
            if not episode_dfs:
                continue
            fans = np.array([df["fan"].values for df in episode_dfs])
            mean_fan = fans.mean(axis=0)
            t = episode_dfs[0]["t"].values
            plt.plot(t, mean_fan, label=tag, color=colors.get(tag, None), linewidth=2)
        
        plt.xlabel("Time (s)")
        plt.ylabel("Fan Speed (%)")
        plt.title(f"{scen}: Controller Comparison (Fan Speed)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(comp_dir / f"{scen}_fan_comparison.png", dpi=150)
        plt.close()


def plot_scenario_grid_per_controller(output_dir: Path, episodes_dir: Path, controllers: list, scenarios: list, temp_warning: float, temp_critical: float):
    """
    For each controller, create a grid of all scenarios (3x4 subplots).
    Saves to output_dir/scenario_grids/
    """
    grid_dir = output_dir / "scenario_grids"
    grid_dir.mkdir(exist_ok=True)
    
    for tag in controllers:
        # Temperature grid
        fig, axes = plt.subplots(3, 4, figsize=(16, 10))
        axes = axes.flatten()
        
        for i, scen in enumerate(scenarios):
            ax = axes[i]
            episode_dfs = [pd.read_csv(f) for f in sorted(episodes_dir.glob(f"{tag}_{scen}_ep*.csv"))]
            if episode_dfs:
                temps = np.array([df["temp"].values for df in episode_dfs])
                mean_temp = temps.mean(axis=0)
                t = episode_dfs[0]["t"].values
                ax.plot(t, mean_temp, color="tab:blue", linewidth=1.5)
                ax.axhline(temp_warning, color="orange", linestyle="--", linewidth=1, alpha=0.7)
                ax.axhline(temp_critical, color="red", linestyle=":", linewidth=1, alpha=0.7)
                ax.set_title(scen, fontsize=9)
                ax.grid(True, alpha=0.3)
                if i >= 8:
                    ax.set_xlabel("Time (s)", fontsize=8)
                if i % 4 == 0:
                    ax.set_ylabel("Temp (°C)", fontsize=8)
        
        plt.suptitle(f"{tag}: Temperature Across All Scenarios", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(grid_dir / f"{tag}_temp_grid.png", dpi=150)
        plt.close()
        
        # Fan grid
        fig, axes = plt.subplots(3, 4, figsize=(16, 10))
        axes = axes.flatten()
        
        for i, scen in enumerate(scenarios):
            ax = axes[i]
            episode_dfs = [pd.read_csv(f) for f in sorted(episodes_dir.glob(f"{tag}_{scen}_ep*.csv"))]
            if episode_dfs:
                fans = np.array([df["fan"].values for df in episode_dfs])
                mean_fan = fans.mean(axis=0)
                t = episode_dfs[0]["t"].values
                ax.plot(t, mean_fan, color="tab:green", linewidth=1.5)
                ax.set_title(scen, fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.set_ylim(15, 105)
                if i >= 8:
                    ax.set_xlabel("Time (s)", fontsize=8)
                if i % 4 == 0:
                    ax.set_ylabel("Fan (%)", fontsize=8)
        
        plt.suptitle(f"{tag}: Fan Speed Across All Scenarios", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(grid_dir / f"{tag}_fan_grid.png", dpi=150)
        plt.close()


def plot_heatmaps(output_dir: Path, combined_df: pd.DataFrame):
    """
    Create heatmaps for controller × scenario metrics.
    Saves to output_dir/heatmaps/
    """
    heatmap_dir = output_dir / "heatmaps"
    heatmap_dir.mkdir(exist_ok=True)
    
    for metric, title in [("mean_fan", "Mean Fan Speed (%)"), 
                          ("cumulative_reward", "Mean Cumulative Reward"),
                          ("violations_warning_entries", "Warning Violations (entries)")]:
        if metric not in combined_df.columns:
            continue
        
        pivot = combined_df.pivot(index="controller", columns="scenario", values=metric)
        
        plt.figure(figsize=(14, 4))
        import seaborn as sns
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn_r" if "violation" in metric else "RdYlGn", 
                    cbar_kws={'label': title}, linewidths=0.5)
        plt.title(f"Heatmap: {title}")
        plt.xlabel("Scenario")
        plt.ylabel("Controller")
        plt.tight_layout()
        plt.savefig(heatmap_dir / f"heatmap_{metric}.png", dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate MPC and RL policies across RC/RCNN")
    parser.add_argument("--eval", nargs="+", choices=["mpc_rc", "mpc_rcnn", "rl_rc", "rl_rcnn", "rl_rc_on_rcnn", "rl_rcnn_on_rc"], required=True,
                        help="Controllers to evaluate")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per scenario")
    parser.add_argument("--horizon", type=int, default=10, help="MPC horizon")
    parser.add_argument("--output-dir", type=str, default="results/policy_eval", help="Output dir")
    parser.add_argument("--plot-mode", type=str, default="none", 
                        choices=["all", "combined", "minimal", "none"],
                        help="Plot generation mode: 'all' (960 individual + combined), 'combined' (only summary plots), 'minimal' (key scenarios only), 'none' (no plots, use plot_results.py later)")
    parser.add_argument(
        "--scenarios", nargs="+", choices=["nominal", "stress", "all"], default=None,
        help="Scenario groups to evaluate. Default: all (both nominal and stress). Allow multiple."
    )
    # Robustness flags (R1): domain shift and actuation delay
    parser.add_argument("--ambient-shift", type=float, default=0.0, help="Additive shift (C) to ambient schedule at eval time")
    parser.add_argument("--power-mult", type=float, default=1.0, help="Multiplicative factor to power schedule at eval time")
    parser.add_argument("--action-delay", type=int, default=0, help="Action delay in steps (>=0). If >0, env applies previous action.")

    # Surrogate config for RC
    parser.add_argument("--rc-C", type=float, default=100.0)
    parser.add_argument("--rc-h", type=float, default=0.05)
    parser.add_argument("--rc-beta", type=float, default=-0.03)
    parser.add_argument("--rc-gamma", type=float, default=0.01)

    # RCNN bundle
    parser.add_argument("--rcnn-bundle", type=str, default=None, help="Path to RC+NN bundle (joblib)")

    # RL models and vecnorms
    parser.add_argument("--rl-rc-model", type=str, default=None)
    parser.add_argument("--rl-rc-vecnorm", type=str, default=None)
    parser.add_argument("--rl-rcnn-model", type=str, default=None)
    parser.add_argument("--rl-rcnn-vecnorm", type=str, default=None)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    (output_dir / "episodes").mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)

    # Environment base config
    env_cfg = {
        "max_steps": 300,
        "temp_warning": 80.0,
        "temp_critical": 90.0,
        "temp_target": 65.0,
        "initial_temp_range": (60.0, 80.0),
        "ambient_range": (20.0, 30.0),
        "power_range": (100.0, 300.0),
        "reward_weights": {"thermal":10.0, "energy":0.1, "oscillation":1.0, "headroom":2.0},
    }

    # Surrogate cfgs
    rc_cfg = {
        "thermal_capacity": args.rc_C,
        "heat_transfer_coeff": args.rc_h,
        "cooling_effectiveness": args.rc_beta,
        "power_to_heat": args.rc_gamma,
        "dt": 1.0,
    }
    rcnn_cfg = {"bundle_path": args.rcnn_bundle} if args.rcnn_bundle else None

    # Controllers to build
    controllers = []  # list of (tag, controller_type, policy_env, policy, base_env)

    if "mpc_rc" in args.eval:
        env_rc = make_env("rc", rc_cfg, env_cfg)
        mpc = MPCController(surrogate=env_rc.surrogate, horizon=args.horizon,
                            temp_target=env_cfg["temp_target"], temp_max=env_cfg["temp_critical"],
                            fan_min=20.0, fan_max=100.0, max_fan_delta=20.0,
                            weight_temp=100.0, weight_effort=0.01, weight_rate=0.5)
        mpc = mpc_act_shim(mpc)
        controllers.append(("mpc_rc", "mpc", env_rc, mpc, env_rc))

    if "mpc_rcnn" in args.eval:
        if not rcnn_cfg:
            raise SystemExit("--rcnn-bundle required for mpc_rcnn")
        env_rcnn = make_env("rcnn", rcnn_cfg, env_cfg)
        mpc2 = MPCController(surrogate=env_rcnn.surrogate, horizon=args.horizon,
                             temp_target=env_cfg["temp_target"], temp_max=env_cfg["temp_critical"],
                             fan_min=20.0, fan_max=100.0, max_fan_delta=20.0,
                             weight_temp=100.0, weight_effort=0.01, weight_rate=0.5)
        mpc2 = mpc_act_shim(mpc2)
        controllers.append(("mpc_rcnn", "mpc", env_rcnn, mpc2, env_rcnn))

    if "rl_rc" in args.eval:
        if not args.rl_rc_model or not args.rl_rc_vecnorm:
            raise SystemExit("--rl-rc-model and --rl-rc-vecnorm required for rl_rc")
        base_env_rc = make_env("rc", rc_cfg, env_cfg)
        vec_env_rc = DummyVecEnv([lambda: base_env_rc])
        vec_env_rc = VecNormalize.load(args.rl_rc_vecnorm, vec_env_rc)
        vec_env_rc.training = False; vec_env_rc.norm_reward = False
        pol_rc = SAC.load(args.rl_rc_model, env=vec_env_rc, device="auto")
        controllers.append(("rl_rc", "rl", vec_env_rc, pol_rc, base_env_rc))

    if "rl_rcnn" in args.eval:
        if not args.rl_rcnn_model or not args.rl_rcnn_vecnorm:
            raise SystemExit("--rl-rcnn-model and --rl-rcnn-vecnorm required for rl_rcnn")
        base_env_rcnn = make_env("rcnn", rcnn_cfg, env_cfg)
        vec_env_rcnn = DummyVecEnv([lambda: base_env_rcnn])
        vec_env_rcnn = VecNormalize.load(args.rl_rcnn_vecnorm, vec_env_rcnn)
        vec_env_rcnn.training = False; vec_env_rcnn.norm_reward = False
        pol_rcnn = SAC.load(args.rl_rcnn_model, env=vec_env_rcnn, device="auto")
        controllers.append(("rl_rcnn", "rl", vec_env_rcnn, pol_rcnn, base_env_rcnn))

    # Cross-surrogate generalization modes
    if "rl_rc_on_rcnn" in args.eval:
        if not args.rl_rc_model or not args.rl_rc_vecnorm:
            raise SystemExit("--rl-rc-model and --rl-rc-vecnorm required for rl_rc_on_rcnn")
        if not rcnn_cfg:
            raise SystemExit("--rcnn-bundle required for rl_rc_on_rcnn")
        base_env_rcnn_x = make_env("rcnn", rcnn_cfg, env_cfg)
        vec_env_rcnn_x = DummyVecEnv([lambda: base_env_rcnn_x])
        # Load the VecNormalize stats from the RL-RC run to preserve observation normalization
        vec_env_rcnn_x = VecNormalize.load(args.rl_rc_vecnorm, vec_env_rcnn_x)
        vec_env_rcnn_x.training = False; vec_env_rcnn_x.norm_reward = False
        pol_rc_on_rcnn = SAC.load(args.rl_rc_model, env=vec_env_rcnn_x, device="auto")
        controllers.append(("rl_rc_on_rcnn", "rl", vec_env_rcnn_x, pol_rc_on_rcnn, base_env_rcnn_x))

    if "rl_rcnn_on_rc" in args.eval:
        if not args.rl_rcnn_model or not args.rl_rcnn_vecnorm:
            raise SystemExit("--rl-rcnn-model and --rl-rcnn-vecnorm required for rl_rcnn_on_rc")
        base_env_rc_x = make_env("rc", rc_cfg, env_cfg)
        vec_env_rc_x = DummyVecEnv([lambda: base_env_rc_x])
        # Load the VecNormalize stats from the RL-RCNN run when evaluating on RC
        vec_env_rc_x = VecNormalize.load(args.rl_rcnn_vecnorm, vec_env_rc_x)
        vec_env_rc_x.training = False; vec_env_rc_x.norm_reward = False
        pol_rcnn_on_rc = SAC.load(args.rl_rcnn_model, env=vec_env_rc_x, device="auto")
        controllers.append(("rl_rcnn_on_rc", "rl", vec_env_rc_x, pol_rcnn_on_rc, base_env_rc_x))

    if not controllers:
        raise SystemExit("No controllers to evaluate")

    # Scenarios
    if (args.scenarios is None) or ("all" in args.scenarios):
        scenarios = NOMINAL_SCENARIOS + STRESS_SCENARIOS
    else:
        scenarios = []
        if "nominal" in args.scenarios:
            scenarios += NOMINAL_SCENARIOS
        if "stress" in args.scenarios:
            scenarios += STRESS_SCENARIOS

    all_results = []
    for tag, ctype, policy_env, policy, base_env in controllers:
        print(f"\n=== Evaluating {tag} ===")
        metrics_rows = []
        for scen in scenarios:
            amb_sched, pow_sched = scenario_schedule(scen, base_env.max_steps)
            # Apply robustness shifts (R1)
            amb_sched = amb_sched + float(args.ambient_shift)
            pow_sched = pow_sched * float(args.power_mult)
            for ep in range(args.episodes):
                # Reset MPC controller state (prev_action) between episodes
                if ctype == "mpc" and hasattr(policy, 'reset'):
                    policy.reset()
                df_ep, m = rollout_episode(policy_env, base_env, policy, ctype, base_env.max_steps, amb_sched, pow_sched, seed=ep, scenario=scen,
                                           temp_warning=base_env.temp_warning, temp_critical=base_env.temp_critical, action_delay=int(args.action_delay))
                # Save episode CSV
                ep_csv = output_dir / "episodes" / f"{tag}_{scen}_ep{ep}.csv"
                df_ep.to_csv(ep_csv, index=False)
                # Individual episode plots (only if mode='all')
                if args.plot_mode == "all":
                    plot_episode(df_ep, output_dir / "plots", scen, tag, base_env.temp_warning, base_env.temp_critical, episode=ep)
                # record metrics
                m.update({"controller": tag, "scenario": scen, "episode": ep})
                metrics_rows.append(m)
        # Save summaries per controller
        df_all, agg = summarize_and_save(metrics_rows, output_dir, tag)
        all_results.append((tag, df_all, agg))

    # Combined comparison: per-scenario bar of cumulative_reward and violations
    combined = []
    for tag, df_all, _ in all_results:
        g = df_all.groupby("scenario").agg({
            "cumulative_reward": "mean",
            # Use 'entries' for crossings (fairer for hot-start), while durations remain in controller CSVs
            "violations_warning_entries": "mean",
            "violations_critical_entries": "mean",
            "time_to_safe_band": "mean",
            "mean_fan": "mean",
        }).reset_index()
        g.insert(0, "controller", tag)
        combined.append(g)
    comb_df = pd.concat(combined, ignore_index=True)
    comb_df.to_csv(output_dir / "combined_per_scenario.csv", index=False)

    # Plot combined reward per scenario and other metrics
    for metric, ylabel in [("cumulative_reward", "Mean Cumulative Reward"), ("mean_fan", "Mean Fan %"), ("violations_warning_entries", "Warn Entries"), ("violations_critical_entries", "Critical Entries")]:
        plt.figure(figsize=(12,5))
        scenarios_unique = comb_df["scenario"].unique()
        x = np.arange(len(scenarios_unique))
        width = 0.18
        for i, tag in enumerate([t for t,_,_ in all_results]):
            sub = comb_df[comb_df["controller"] == tag]
            vals = [float(sub[sub["scenario"]==s][metric].values[0]) if s in set(sub["scenario"]) else np.nan for s in scenarios_unique]
            plt.bar(x + i*width, vals, width=width, label=tag)
        plt.xticks(x + width* (len(all_results)-1)/2, scenarios_unique, rotation=30, ha='right')
        plt.ylabel(ylabel)
        plt.title(f"Comparison: {ylabel} by Scenario")
        plt.legend(); plt.tight_layout()
        plt.savefig(output_dir / f"combined_{metric}.png", dpi=150)
        plt.close()

    # Generate combined plots based on plot-mode
    if args.plot_mode != "none":
        print("\n[INFO] Generating combined plots...")
        controller_tags = [t for t, _, _ in all_results]
        
        # Filter scenarios for minimal mode
        if args.plot_mode == "minimal":
            key_scenarios = ["baseline", "thermal_high_start", "combined_extreme", "low_workload"]
            plot_scenarios = [s for s in scenarios if s in key_scenarios]
            print(f"  [Minimal mode: plotting only {len(plot_scenarios)} key scenarios: {plot_scenarios}]")
        else:
            plot_scenarios = scenarios
        
        # Multi-episode overlays (skip in minimal mode to reduce clutter)
        if args.plot_mode in ["all", "combined"]:
            print("  - Multi-episode overlays (mean ± std per controller-scenario)...")
            plot_combined_multi_episode(output_dir, output_dir / "episodes", controller_tags, plot_scenarios, 
                                         base_env.temp_warning, base_env.temp_critical)
        
        # Controller comparisons (always generate - most useful)
        print("  - Controller comparisons (all controllers per scenario)...")
        plot_controller_comparison_per_scenario(output_dir, output_dir / "episodes", controller_tags, plot_scenarios,
                                               base_env.temp_warning, base_env.temp_critical)
        
        # Scenario grids (always generate - compact overview)
        print("  - Scenario grids (all scenarios per controller)...")
        plot_scenario_grid_per_controller(output_dir, output_dir / "episodes", controller_tags, scenarios,
                                         base_env.temp_warning, base_env.temp_critical)
        
        # Heatmaps (always generate - best summary)
        print("  - Heatmaps (controller × scenario metrics)...")
        plot_heatmaps(output_dir, comb_df)
        
        print(f"\n[INFO] Saved evaluation to {output_dir}")
        print(f"[INFO] Plot summary (mode={args.plot_mode}):")
        if args.plot_mode == "all":
            print(f"  - Individual episodes: {len(controller_tags) * len(scenarios) * args.episodes * 2} plots (plots/)")
            print(f"  - Multi-episode overlays: {len(controller_tags) * len(plot_scenarios) * 2} plots (combined_episodes/)")
        elif args.plot_mode == "combined":
            print(f"  - Multi-episode overlays: {len(controller_tags) * len(plot_scenarios) * 2} plots (combined_episodes/)")
        print(f"  - Controller comparisons: {len(plot_scenarios) * 2} plots (controller_comparison/)")
        print(f"  - Scenario grids: {len(controller_tags) * 2} plots (scenario_grids/)")
        print(f"  - Heatmaps: 3 plots (heatmaps/)")
    else:
        print(f"\n[INFO] Saved evaluation to {output_dir} (no plots generated, mode=none)")


if __name__ == "__main__":
    main()
