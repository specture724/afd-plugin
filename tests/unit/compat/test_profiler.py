from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from afd_plugin.compat.profiler import (
    afd_gpu_profiler_config,
    create_afd_gpu_profiler,
    step_afd_gpu_profiler,
    stop_afd_gpu_profiler,
)

_ENV_NAMES = (
    "AFD_GPU_ATTENTION_PROFILER_ENABLE",
    "AFD_GPU_ATTENTION_PROFILER_WAIT",
    "AFD_GPU_ATTENTION_PROFILER_WARMUP",
    "AFD_GPU_ATTENTION_PROFILER_ACTIVE",
    "AFD_GPU_ATTENTION_PROFILER_REPEAT",
    "AFD_GPU_ATTENTION_PROFILER_SKIP_FIRST",
    "AFD_GPU_ATTENTION_PROFILER_DIR",
    "AFD_GPU_FFN_PROFILER_ENABLE",
    "AFD_GPU_FFN_PROFILER_WAIT",
    "AFD_GPU_FFN_PROFILER_WARMUP",
    "AFD_GPU_FFN_PROFILER_ACTIVE",
    "AFD_GPU_FFN_PROFILER_REPEAT",
    "AFD_GPU_FFN_PROFILER_SKIP_FIRST",
    "AFD_GPU_FFN_PROFILER_DIR",
    "VLLM_TORCH_PROFILER_DIR",
)


@pytest.fixture(autouse=True)
def _clear_profiler_env(monkeypatch):
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_gpu_profiler_defaults_are_disabled():
    attention = afd_gpu_profiler_config("attention")
    ffn = afd_gpu_profiler_config("ffn")

    assert attention.enabled is False
    assert attention.wait == 2500
    assert attention.warmup == 1
    assert attention.active == 10
    assert attention.repeat == 1
    assert attention.skip_first == 0
    assert attention.trace_dir == "./profiler_logs/attn"
    assert ffn.enabled is False
    assert ffn.trace_dir == "./profiler_logs/ffn"


def test_gpu_profiler_dir_falls_back_to_vllm_torch_profiler_dir(monkeypatch):
    monkeypatch.setenv("VLLM_TORCH_PROFILER_DIR", "/tmp/vllm-profile")

    assert afd_gpu_profiler_config("attention").trace_dir == "/tmp/vllm-profile"

    monkeypatch.setenv("AFD_GPU_ATTENTION_PROFILER_DIR", "/tmp/afd-attn")

    assert afd_gpu_profiler_config("attention").trace_dir == "/tmp/afd-attn"


def test_create_gpu_profiler_uses_configured_schedule(monkeypatch):
    profiler_module = _FakeTorchProfiler()
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(profiler=profiler_module),
    )
    monkeypatch.setenv("AFD_GPU_FFN_PROFILER_ENABLE", "true")
    monkeypatch.setenv("AFD_GPU_FFN_PROFILER_WAIT", "3")
    monkeypatch.setenv("AFD_GPU_FFN_PROFILER_WARMUP", "4")
    monkeypatch.setenv("AFD_GPU_FFN_PROFILER_ACTIVE", "5")
    monkeypatch.setenv("AFD_GPU_FFN_PROFILER_REPEAT", "6")
    monkeypatch.setenv("AFD_GPU_FFN_PROFILER_SKIP_FIRST", "7")
    monkeypatch.setenv("AFD_GPU_FFN_PROFILER_DIR", "/tmp/afd-ffn")

    profiler = create_afd_gpu_profiler("ffn")

    assert profiler is profiler_module.created_profiler
    assert profiler.started is True
    assert profiler_module.schedule_kwargs == {
        "wait": 3,
        "warmup": 4,
        "active": 5,
        "repeat": 6,
        "skip_first": 7,
    }
    assert profiler_module.profile_kwargs["record_shapes"] is True
    assert profiler_module.profile_kwargs["profile_memory"] is False
    assert profiler_module.profile_kwargs["with_stack"] is False
    assert profiler_module.trace_dir == "/tmp/afd-ffn"


def test_step_gpu_profiler_ignores_disabled_profiler():
    step_afd_gpu_profiler(None)

    profiler = _StepProfiler()
    step_afd_gpu_profiler(profiler)

    assert profiler.steps == 1


def test_stop_gpu_profiler_ignores_disabled_profiler():
    stop_afd_gpu_profiler(None)

    profiler = _StepProfiler()
    stop_afd_gpu_profiler(profiler)

    assert profiler.stopped is True


class _StepProfiler:
    def __init__(self):
        self.steps = 0
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def step(self):
        self.steps += 1


class _FakeTorchProfiler:
    class ProfilerActivity:
        CPU = "cpu"
        CUDA = "cuda"

    def __init__(self):
        self.created_profiler = _StepProfiler()
        self.schedule_kwargs = None
        self.profile_kwargs = None
        self.trace_dir = None

    def schedule(self, **kwargs):
        self.schedule_kwargs = kwargs
        return kwargs

    def tensorboard_trace_handler(self, trace_dir):
        self.trace_dir = trace_dir
        return ("handler", trace_dir)

    def profile(self, **kwargs):
        self.profile_kwargs = kwargs
        return self.created_profiler


def test_window_mode_records_one_window_and_exports_a_chrome_trace(
    monkeypatch, tmp_path
):
    # The window is what lets processes start and stop on the same wall clock.
    # A thread drives it because the Attention runner steps too rarely at long
    # prefill steps for step-driven start/stop to hit a few-second window.
    import json
    import time

    import pytest

    torch = pytest.importorskip("torch")
    from afd_plugin.compat import profiler as profiler_module

    window = tmp_path / "window"
    monkeypatch.setenv("AFD_GPU_FFN_PROFILER_ENABLE", "1")
    monkeypatch.setenv("AFD_GPU_FFN_PROFILER_DIR", str(tmp_path / "traces"))
    monkeypatch.setenv("AFD_GPU_PROFILER_WINDOW_FILE", str(window))
    # A huge poll interval parks the thread so the test drives poll() itself.
    monkeypatch.setattr(profiler_module, "_WINDOW_POLL_S", 3600.0)

    prof = profiler_module.create_afd_gpu_profiler("ffn")
    assert prof is not None

    prof.poll()  # no window file yet: nothing starts
    assert prof._profiler is None

    now = time.time()
    window.write_text(f"{now - 1} {now + 60} armX")
    prof.poll()
    assert prof._profiler is not None
    torch.ones(4) * 2  # something for the trace to hold
    prof.step()  # a no-op in window mode: the thread owns start/stop
    assert prof._profiler is not None

    window.write_text(f"{now - 1} {now - 0.5} armX")  # window has closed
    prof.poll()
    assert prof._profiler is None

    traces = list((tmp_path / "traces").glob("armX_ffn_*.json"))
    assert len(traces) == 1
    assert "traceEvents" in json.loads(traces[0].read_text())

    # One window per process: reopening it does not record a second trace.
    window.write_text(f"{time.time() - 1} {time.time() + 60} armX")
    prof.poll()
    assert prof._profiler is None
