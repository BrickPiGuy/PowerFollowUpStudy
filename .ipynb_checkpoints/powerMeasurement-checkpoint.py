# pip install nvidia-ml-py  # (or: pip install pynvml)
import time, subprocess
import pynvml as nvml

def _read_energy_mJ(handle):
    """
    Try energy-counter APIs; raise AttributeError if none exist.
    """
    # Some recent NVMLs also expose a _v2 variant; try that first if present.
    if hasattr(nvml, "nvmlDeviceGetTotalEnergyConsumption_v2"):
        return nvml.nvmlDeviceGetTotalEnergyConsumption_v2(handle)
    if hasattr(nvml, "nvmlDeviceGetTotalEnergyConsumption"):
        return nvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    raise AttributeError("Energy-counter API not available in your pynvml/NVML")

def _integrate_power_joules(handle, proc, poll_s=0.05):
    """
    Fallback: sample nvmlDeviceGetPowerUsage (mW) and integrate (trapezoid rule).
    """
    # Initial sample
    start = time.perf_counter()
    last_t = start
    last_p_w = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW -> W
    energy_j = 0.0

    # Sample while the process runs
    while proc.poll() is None:
        time.sleep(poll_s)
        t = time.perf_counter()
        p_w = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        dt = t - last_t
        energy_j += 0.5 * (p_w + last_p_w) * dt  # trapezoidal area (J)
        last_t, last_p_w = t, p_w

    # One final sample at exit
    t = time.perf_counter()
    p_w = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
    dt = t - last_t
    energy_j += 0.5 * (p_w + last_p_w) * dt

    elapsed = t - start
    return energy_j, elapsed

def measure_gpu_energy(cmd, gpu_index=0, poll_s=0.05):
    nvml.nvmlInit()
    try:
        h = nvml.nvmlDeviceGetHandleByIndex(gpu_index)

        # Prefer the hardware energy counter if available and supported
        try:
            e0_mJ = _read_energy_mJ(h)
            t0 = time.perf_counter()
            subprocess.run(cmd, check=True)
            e1_mJ = _read_energy_mJ(h)
            t1 = time.perf_counter()

            joules = (e1_mJ - e0_mJ) / 1000.0  # mJ -> J
            elapsed = t1 - t0
            avg_watts = joules / elapsed if elapsed > 0 else float("nan")
            mode = "NVML energy counter"
        except (AttributeError, nvml.NVMLError_FunctionNotFound, nvml.NVMLError_NotSupported):
            # Fallback to power sampling + integration
            proc = subprocess.Popen(cmd)
            joules, elapsed = _integrate_power_joules(h, proc, poll_s=poll_s)
            avg_watts = joules / elapsed if elapsed > 0 else float("nan")
            mode = "Power sampling (integrated)"

        return joules, avg_watts, elapsed, mode
    finally:
        nvml.nvmlShutdown()

if __name__ == "__main__":
    cmd = ["python", "benchmarkScript.py"]
    joules, avg_watts, seconds, mode = measure_gpu_energy(cmd, gpu_index=0, poll_s=0.05)
    print(f"[{mode}] Energy: {joules:.2f} J, Avg power: {avg_watts:.2f} W, Elapsed: {seconds:.2f} s")
