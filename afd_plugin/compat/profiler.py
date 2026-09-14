# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Plugin-owned CUDA profiler helpers for AFD GPU runners."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

AFDGPUProfilerRole = Literal["attention", "ffn"]

_ENV_PREFIX: Final[dict[AFDGPUProfilerRole, str]] = {
    "attention": "AFD_GPU_ATTENTION_PROFILER",
    "ffn": "AFD_GPU_FFN_PROFILER",
}
_DEFAULT_DIR: Final[dict[AFDGPUProfilerRole, str]] = {
    "attention": "./profiler_logs/attn",
    "ffn": "./profiler_logs/ffn",
}
_DEFAULT_WAIT_STEPS: Final[int] = 2500
_DEFAULT_WARMUP_STEPS: Final[int] = 1
_DEFAULT_ACTIVE_STEPS: Final[int] = 10
_DEFAULT_REPEAT: Final[int] = 1
_DEFAULT_SKIP_FIRST_STEPS: Final[int] = 0
_VLLM_TORCH_PROFILER_DIR_ENV: Final[str] = "VLLM_TORCH_PROFILER_DIR"

# Wall-clock window mode. The step schedule cannot line the two roles up: the
# Attention runner steps once per scheduler step, but the connector-driven FFN
# runner steps on every receive poll, idle ones included, so its wait/active
# counts are consumed by spin long before traffic arrives. A window file holding
# "<start_epoch> <stop_epoch> <tag>" starts and stops every process on the same
# wall clock instead, which is also what lets their traces be merged.
_WINDOW_FILE_ENV: Final[str] = "AFD_GPU_PROFILER_WINDOW_FILE"
_WINDOW_POLL_S: Final[float] = 0.05


@dataclass(frozen=True)
class AFDGPUProfilerConfig:
    enabled: bool
    wait: int
    warmup: int
    active: int
    repeat: int
    skip_first: int
    trace_dir: str


def afd_gpu_profiler_config(role: AFDGPUProfilerRole) -> AFDGPUProfilerConfig:
    """Read plugin-owned profiler settings for an AFD GPU runner role."""

    prefix = _ENV_PREFIX[role]
    return AFDGPUProfilerConfig(
        enabled=_env_bool(f"{prefix}_ENABLE", default=False),
        wait=_env_int(f"{prefix}_WAIT", default=_DEFAULT_WAIT_STEPS),
        warmup=_env_int(f"{prefix}_WARMUP", default=_DEFAULT_WARMUP_STEPS),
        active=_env_int(f"{prefix}_ACTIVE", default=_DEFAULT_ACTIVE_STEPS),
        repeat=_env_int(f"{prefix}_REPEAT", default=_DEFAULT_REPEAT),
        skip_first=_env_int(
            f"{prefix}_SKIP_FIRST",
            default=_DEFAULT_SKIP_FIRST_STEPS,
        ),
        trace_dir=_env_dir(f"{prefix}_DIR", default=_DEFAULT_DIR[role]),
    )


class _WindowedProfiler:
    """Record exactly one wall-clock window, then export a Chrome trace.

    A daemon thread owns start and stop. Driving them from the runners' step
    calls does not work for the Attention role: it steps once per scheduler
    step, which at 8192-token prefill is about a second, so a few-second window
    opened and closed on step boundaries and recorded one step or none. CUPTI
    and the op observers are process-wide, so starting the profiler from a side
    thread still records every thread's CPU ops and every stream's kernels.

    Duck-types the two methods the runners call on a ``torch.profiler.profile``
    so the call sites need no change. The window is read from a file rather than
    from the environment because it has to be set after the servers are up and
    warm, which is long after the environment was fixed.
    """

    def __init__(self, role: AFDGPUProfilerRole, trace_dir: str, window_file: str):
        import threading

        import torch

        self._role = role
        self._trace_dir = trace_dir
        self._window_file = window_file
        self._profiler: torch.profiler.profile | None = None
        self._tag = ""
        self._done = False
        self._lock = threading.Lock()
        # Captured here, on the runner's thread, where the device is bound; the
        # window thread would otherwise see device 0.
        self._device = torch.cuda.current_device() if torch.cuda.is_available() else -1
        self._thread = threading.Thread(
            target=self._run,
            name=f"afd-{role}-profiler-window",
            daemon=True,
        )
        self._thread.start()

    def _read_window(self) -> tuple[float, float, str] | None:
        try:
            with open(self._window_file) as handle:
                start, stop, tag = handle.read().split()[:3]
            return float(start), float(stop), tag
        except (OSError, ValueError):
            return None

    def _run(self) -> None:
        while not self._done:
            self.poll()
            time.sleep(_WINDOW_POLL_S)

    def poll(self) -> None:
        """Start or stop against the window file; the thread calls this."""
        with self._lock:
            if self._done:
                return
            window = self._read_window()
            if window is None:
                return
            start, stop, tag = window
            wall = time.time()
            if self._profiler is None and start <= wall < stop:
                import torch

                self._tag = tag
                self._profiler = torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    record_shapes=False,
                    profile_memory=False,
                    with_stack=False,
                )
                self._profiler.start()
                logger.warning("AFD %s profiler window %s started", self._role, tag)
            elif self._profiler is not None and wall >= stop:
                self._export()

    def step(self) -> None:
        # The window thread owns start and stop.
        return

    def stop(self) -> None:
        with self._lock:
            if self._profiler is not None:
                self._export()
            self._done = True

    def _export(self) -> None:
        assert self._profiler is not None
        self._profiler.stop()
        os.makedirs(self._trace_dir, exist_ok=True)
        path = os.path.join(
            self._trace_dir,
            f"{self._tag}_{self._role}_cuda{self._device}_pid{os.getpid()}.json",
        )
        self._profiler.export_chrome_trace(path)
        logger.warning(
            "AFD %s profiler window %s wrote %s", self._role, self._tag, path
        )
        self._profiler = None
        self._done = True


def create_afd_gpu_profiler(role: AFDGPUProfilerRole) -> torch.profiler.profile | None:
    """Create a torch profiler when the plugin-owned env enables it."""

    config = afd_gpu_profiler_config(role)
    if not config.enabled:
        return None

    window_file = os.getenv(_WINDOW_FILE_ENV)
    if window_file:
        logger.warning(
            "AFD GPU %s profiler in window mode: %s -> %s",
            role,
            window_file,
            config.trace_dir,
        )
        return _WindowedProfiler(role, config.trace_dir, window_file)  # type: ignore[return-value]

    import torch

    logger.info(
        "AFD GPU %s profiler enabled. Traces will be saved to: %s",
        role,
        config.trace_dir,
    )
    profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=config.wait,
            warmup=config.warmup,
            active=config.active,
            repeat=config.repeat,
            skip_first=config.skip_first,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(config.trace_dir),
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
    )
    profiler.start()
    return profiler


def step_afd_gpu_profiler(profiler: torch.profiler.profile | None) -> None:
    if profiler is not None:
        profiler.step()


def stop_afd_gpu_profiler(profiler: torch.profiler.profile | None) -> None:
    if profiler is not None:
        profiler.stop()


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}")


def _env_int(name: str, *, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _env_dir(name: str, *, default: str) -> str:
    return os.getenv(name) or os.getenv(_VLLM_TORCH_PROFILER_DIR_ENV) or default


__all__ = [
    "AFDGPUProfilerConfig",
    "afd_gpu_profiler_config",
    "create_afd_gpu_profiler",
    "step_afd_gpu_profiler",
    "stop_afd_gpu_profiler",
]
