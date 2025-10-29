# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive play script for the Agibot Toy2Box placing task.

This script launches the :code:`Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0`
environment and wires a SE(3) teleoperation device so you can manually control
Agibot's right arm with RMPflow.

Example usage::

    ./isaaclab.sh -p scripts/environments/play_toy2box_agibot.py --teleop_device spacemouse

By default the script creates a single environment, enables relative pose
commands, and maps the ``R`` key (or controller reset button) to reset the
scenario. Pass ``--num_envs`` to spawn multiple copies or ``--absolute_mode`` to
switch to absolute end-effector control.
"""

"""Launch Isaac Sim Simulator first."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable

from isaaclab.app import AppLauncher


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Play the Agibot Toy2Box task via SE(3) teleoperation.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0",
    help="Name of the task to launch.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    help="Teleoperation device to use (keyboard, spacemouse, gamepad).",
)
parser.add_argument(
    "--sensitivity",
    type=float,
    default=1.0,
    help="Scale factor applied to translational and rotational teleop commands.",
)
parser.add_argument(
    "--absolute_mode",
    action="store_true",
    default=False,
    help="Use absolute end-effector commands instead of the default relative mode.",
)
parser.add_argument(
    "--reset_key",
    type=str,
    default="R",
    help="Key or button binding used to reset the environment when pressed.",
)
# Append AppLauncher CLI args (device, headless, experience, etc.)
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments before importing heavy modules
args_cli = parser.parse_args()

# Set the relative mode flag expected by the env configuration before it is parsed.
os.environ["USE_RELATIVE_MODE"] = "False" if args_cli.absolute_mode else "True"

# Launch Omniverse
app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import omni.log

from isaaclab.devices import Se3Gamepad, Se3GamepadCfg, Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.managers import TerminationTermCfg as DoneTerm

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.utils import parse_env_cfg

# PLACEHOLDER: Extension template (do not remove this comment)


def _build_fallback_device(
    teleop_device: str,
    sensitivity: float,
    callbacks: dict[str, Callable[[], None]],
) -> Se3Keyboard | Se3SpaceMouse | Se3Gamepad:
    """Create a fallback teleoperation device when the env configuration does not provide one."""
    sensitivity = max(sensitivity, 1e-3)
    device_name = teleop_device.lower()
    if device_name == "keyboard":
        interface = Se3Keyboard(
            Se3KeyboardCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity)
        )
    elif device_name == "spacemouse":
        interface = Se3SpaceMouse(
            Se3SpaceMouseCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity)
        )
    elif device_name == "gamepad":
        interface = Se3Gamepad(
            Se3GamepadCfg(pos_sensitivity=0.1 * sensitivity, rot_sensitivity=0.1 * sensitivity)
        )
    else:
        raise ValueError(
            f"Unsupported teleop device '{teleop_device}'. Supported options: keyboard, spacemouse, gamepad."
        )

    for key, callback in callbacks.items():
        try:
            interface.add_callback(key, callback)
        except (ValueError, TypeError) as exc:
            omni.log.warn(f"Failed to register callback for '{key}' on {teleop_device}: {exc}")

    return interface


def _create_teleop_interface(env_cfg, teleop_device: str, sensitivity: float, callbacks: dict[str, Callable[[], None]]):
    """Instantiate the teleoperation interface, preferring the env configuration."""
    if hasattr(env_cfg, "teleop_devices") and teleop_device in getattr(env_cfg.teleop_devices, "devices", {}):
        try:
            return create_teleop_device(teleop_device, env_cfg.teleop_devices.devices, callbacks)
        except Exception as exc:  # pragma: no cover - safety net around external factory
            omni.log.warn(f"Failed to build teleop device from env config: {exc}. Falling back to default constructors.")
    else:
        omni.log.warn(
            f"No teleop device '{teleop_device}' registered in env config. Falling back to default constructors."
        )
    return _build_fallback_device(teleop_device, sensitivity, callbacks)


def _configure_env():
    """Parse and tweak the environment configuration for interactive play."""
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.env_name = args_cli.task
    # Allow manual resets by disabling automatic timeout termination
    if hasattr(env_cfg, "terminations") and hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None
    # For lift-style tasks make sure there is a goal reached condition so reset works
    if "Lift" in args_cli.task:
        env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)
    return env_cfg


def main():  # noqa: D401 - short docstring provided below
    """Launch the environment and run a teleoperation loop."""
    env_cfg = _configure_env()

    try:
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    except Exception as exc:  # pragma: no cover - runtime dependency on simulator
        omni.log.error(f"Failed to create environment '{args_cli.task}': {exc}")
        simulation_app.close()
        return

    reset_requested = False
    teleop_active = True

    def _schedule_reset():
        nonlocal reset_requested
        reset_requested = True
        print("Reset triggered - the environment will reset after the current frame.")

    def _set_active(active: bool):
        nonlocal teleop_active
        teleop_active = active
        state = "resumed" if teleop_active else "paused"
        print(f"Teleoperation {state}.")

    callbacks: dict[str, Callable[[], None]] = {
        args_cli.reset_key.upper(): _schedule_reset,
        "RESET": _schedule_reset,
        "START": lambda: _set_active(True),
        "STOP": lambda: _set_active(False),
    }

    teleop_interface = _create_teleop_interface(env_cfg, args_cli.teleop_device, args_cli.sensitivity, callbacks)
    teleop_interface.reset()

    observations, _ = env.reset()
    print(f"Loaded task: {args_cli.task}")
    print(f"Observation keys: {list(observations.keys())}")
    print(
        "Controls: use the teleop device to move the gripper, press"
        f" {args_cli.reset_key.upper()} to reset, and START/STOP to pause/resume when supported."
    )

    while simulation_app.is_running():
        with torch.inference_mode():
            try:
                action = teleop_interface.advance()
            except Exception as exc:  # pragma: no cover - runtime specific handling
                omni.log.error(f"Teleop device error: {exc}")
                break

            if teleop_active:
                actions = action.repeat(env.num_envs, 1)
                _, _, terminated, truncated, _ = env.step(actions)
            else:
                env.sim.render()
                terminated = truncated = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

            if reset_requested or bool(torch.any(terminated | truncated)):
                env.reset()
                teleop_interface.reset()
                reset_requested = False

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
