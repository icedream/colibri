#!/usr/bin/env python3
"""Hardware and model placement planning for colibri's disk/RAM/VRAM tiers."""

import json
import os
import re
import shutil
import statistics
import subprocess
from pathlib import Path


GB = 1_000_000_000
EXPERT_RE = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.")


def _tensor_sizes(path):
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError(f"short safetensors header: {path}")
        length = int.from_bytes(raw, "little")
        if length < 2 or length > file_size - 8:
            raise ValueError(f"invalid safetensors header length: {path}")
        header = json.loads(stream.read(length))
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        if not 0 <= start <= end <= file_size - 8 - length:
            raise ValueError(f"invalid tensor offsets for {name}: {path}")
        yield name, end - start


def analyze_model(model):
    model = Path(model).resolve()
    config_path = model / "config.json"
    if not config_path.is_file():
        raise ValueError(f"missing config.json: {model}")
    config = json.loads(config_path.read_text())
    shards = sorted(model.glob("*.safetensors"))
    if not shards:
        raise ValueError(f"no safetensors shards: {model}")

    dense_bytes = 0
    expert_groups = {}
    for shard in shards:
        for name, size in _tensor_sizes(shard):
            match = EXPERT_RE.search(name)
            if match:
                key = tuple(map(int, match.groups()))
                expert_groups[key] = expert_groups.get(key, 0) + size
            else:
                dense_bytes += size

    layer_sizes = {}
    for (layer, _), size in expert_groups.items():
        layer_sizes.setdefault(layer, []).append(size)
    per_layer = {layer: int(statistics.median(sizes)) for layer, sizes in layer_sizes.items()}
    per_cap_bytes = sum(per_layer.values())
    typical_expert_bytes = int(statistics.median(per_layer.values())) if per_layer else 0
    model_bytes = sum(shard.stat().st_size for shard in shards)
    return {
        "path": str(model),
        "shards": len(shards),
        "model_bytes": model_bytes,
        "dense_bytes": dense_bytes,
        "expert_bytes": sum(expert_groups.values()),
        "expert_count": len(expert_groups),
        "expert_layers": len(per_layer),
        "typical_expert_bytes": typical_expert_bytes,
        "per_cap_bytes": per_cap_bytes,
        "config": config,
    }


def memory_available():
    try:
        text = Path("/proc/meminfo").read_text()
        return int(re.search(r"MemAvailable:\s+(\d+)", text).group(1)) * 1024
    except (OSError, AttributeError):
        return 0


def discover_gpus():
    """Discover GPUs via nvidia-smi (NVIDIA) or rocminfo (AMD/ROCm)."""
    devices = _discover_nvidia()
    if devices:
        return devices
    return _discover_rocm()


def _discover_nvidia():
    command = ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free",
               "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return []
    devices = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 3)]
        if len(fields) != 4:
            continue
        try:
            index, total, free = int(fields[0]), int(fields[2]), int(fields[3])
        except ValueError:
            continue
        devices.append({"index": index, "name": fields[1],
                        "total_bytes": total * 1024 * 1024,
                        "free_bytes": free * 1024 * 1024})
    return devices


def _discover_rocm():
    """Discover AMD GPUs via rocminfo, falling back to /sys/class/drm/ card enumeration."""
    # Try rocminfo first (gives name + memory info)
    devices = _discover_rocm_rocminfo()
    if devices:
        return devices
    # Fallback: count amdgpu cards in /sys/class/drm/
    return _discover_rocm_sysfs()


def _discover_rocm_rocminfo():
    """Parse rocminfo output for GPU devices only (filter out CPUs and ISAs)."""
    try:
        result = subprocess.run(["rocminfo"], text=True, capture_output=True,
                                timeout=5, check=True)
    except (OSError, subprocess.SubprocessError):
        return []

    devices = []
    current_device = None
    device_type = None  # "CPU", "GPU", or None
    pending_name = None  # Name to assign after confirming device type

    for line in result.stdout.splitlines():
        stripped = line.strip()

        # Detect agent start: "Agent N"
        if stripped.startswith("Agent "):
            # Save previous device if it was a GPU
            if current_device is not None and device_type == "GPU":
                devices.append(current_device)
            current_device = {"index": len(devices), "name": "Unknown", "total_bytes": 0, "free_bytes": 0}
            device_type = None
            pending_name = None
            continue

        # Detect device type
        if "Device Type:" in stripped:
            if "CPU" in stripped:
                device_type = "CPU"
            elif "GPU" in stripped:
                device_type = "GPU"
            # If we have a pending name and now know it's a GPU, assign it
            if device_type == "GPU" and pending_name and current_device:
                current_device["name"] = pending_name
                pending_name = None
            continue

        # Capture name before device type is known (device names come first in rocminfo)
        if (stripped.startswith("Name:") and 
            "Marketing Name:" not in stripped and 
            pending_name is None and 
            len(line) - len(line.lstrip()) < 5):
            pending_name = stripped.split("Name:", 1)[1].strip()

        # Extract memory from Pool sections (first GLOBAL pool = VRAM)
        if device_type == "GPU" and current_device is not None and current_device["total_bytes"] == 0:
            if stripped.startswith("Size:"):
                try:
                    size_str = stripped.split("Size:", 1)[1].strip()
                    # rocminfo format: "25149440(0x17fc000) KB" -> extract decimal part
                    if "(" in size_str:
                        size_str = size_str.split("(")[0].strip()
                    if size_str.startswith("0x"):
                        mem_kb = int(size_str, 16)
                    else:
                        mem_kb = int(size_str.replace(",", "").strip())
                    # rocminfo reports in KB, convert to bytes
                    mem_bytes = mem_kb * 1024
                    current_device["total_bytes"] = mem_bytes
                    current_device["free_bytes"] = mem_bytes
                except (ValueError, IndexError):
                    pass

    if current_device is not None and device_type == "GPU":
        # Assign pending name if not yet assigned
        if pending_name and current_device["name"] == "Unknown":
            current_device["name"] = pending_name
        devices.append(current_device)

    return devices


def _discover_rocm_sysfs():
    """Enumerate AMD GPUs from /sys/class/drm/ card entries."""
    devices = []
    drm_path = Path("/sys/class/drm")
    if not drm_path.exists():
        return devices

    card_index = 0
    for card_dir in sorted(drm_path.iterdir()):
        if not card_dir.name.startswith("card"):
            continue
        if not card_dir.is_symlink():
            continue
        device_path = card_dir / "device"
        if not device_path.exists():
            continue
        # Check if it's an AMD GPU (vendor 0x1002)
        vendor_file = device_path / "vendor"
        if not vendor_file.exists():
            continue
        try:
            vendor = int(vendor_file.read_text().strip(), 16)
            if vendor != 0x1002:  # AMD vendor ID
                continue
        except (ValueError, OSError):
            continue

        # Try to read VRAM from memory_info_vram_total (kernel 5.15+)
        total_bytes = 0
        vram_file = device_path / "memory_info_vram_total"
        if vram_file.exists():
            try:
                total_bytes = int(vram_file.read_text().strip())
            except (ValueError, OSError):
                pass

        # Fallback: estimate from device ID or use a default
        if total_bytes == 0:
            # Could read device_id and look up in a table, but for now use 0
            # The user can override with --vram
            total_bytes = 0

        devices.append({
            "index": card_index,
            "name": f"AMD GPU (card{card_dir.name.split('card')[1]})",
            "total_bytes": total_bytes,
            "free_bytes": total_bytes,  # unknown without rocminfo
        })
        card_index += 1

    return devices


def build_plan(model, ram_gb=0, context=4096, gpu_indices=None, vram_gb=0,
               available_memory=None, available_disk=None, gpus=None):
    info = analyze_model(model)
    cfg = info["config"]
    available_memory = memory_available() if available_memory is None else available_memory
    if available_disk is None:
        try:
            usage = shutil.disk_usage(info["path"])
            available_disk = usage.free
        except OSError:
            available_disk = 500 * GB
    gpus = discover_gpus() if gpus is None else gpus
    if gpu_indices is not None:
        wanted = set(gpu_indices)
        gpus = [gpu for gpu in gpus if gpu["index"] in wanted]

    ram_budget = int(ram_gb * GB) if ram_gb > 0 else int(available_memory * 0.88)
    if ram_budget < 4 * GB:
        ram_budget = 8 * GB
    typical = info["typical_expert_bytes"]
    layers = int(cfg.get("num_hidden_layers", 0)) + 1
    kv_bytes = layers * context * (int(cfg.get("kv_lora_rank", 0)) +
                                   int(cfg.get("qk_rope_head_dim", 0))) * 4
    kv_buffer = context * int(cfg.get("num_attention_heads", 0)) * (
        int(cfg.get("qk_nope_head_dim", 0)) + int(cfg.get("v_head_dim", 0))) * 4
    runtime_bytes = int(1.2 * GB + 2.5 * GB + 64 * typical + kv_bytes + kv_buffer)
    cache_bytes = max(0, ram_budget - info["dense_bytes"] - runtime_bytes)
    per_cap = info["per_cap_bytes"]
    configured_experts = int(cfg.get("n_routed_experts", 0))
    cap = int(cache_bytes // per_cap) if per_cap else 0
    if configured_experts:
        cap = min(cap, configured_experts)

    reserve = 2 * GB
    gpu_plan = []
    safe_vram = 0
    for gpu in gpus:
        usable = max(0, gpu["free_bytes"] - reserve)
        safe_vram += usable
        gpu_plan.append(dict(gpu, reserve_bytes=reserve, usable_bytes=usable))
    requested_vram = int(vram_gb * GB) if vram_gb > 0 else safe_vram
    vram_budget = min(requested_vram, safe_vram, cache_bytes)
    vram_experts = int(vram_budget // typical) if typical else 0

    warnings = []
    if cap < 1:
        warnings.append("RAM budget cannot hold one expert slot per sparse layer")
    if gpu_indices is not None and len(gpus) != len(set(gpu_indices)):
        warnings.append("one or more requested GPUs were not detected")
    if gpus and vram_budget < requested_vram:
        warnings.append("VRAM tier was clamped by free VRAM or its required RAM backing")

    return {
        "version": 1,
        "model": {key: value for key, value in info.items() if key != "config"},
        "tiers": {
            "disk": {"role": "backing", "model_bytes": info["model_bytes"],
                     "available_bytes": available_disk},
            "ram": {"role": "resident+cache", "available_bytes": available_memory,
                    "budget_bytes": ram_budget, "dense_bytes": info["dense_bytes"],
                    "runtime_bytes": runtime_bytes, "expert_cache_bytes": cache_bytes,
                    "cache_slots_per_layer": cap},
            "vram": {"role": "hot-experts", "devices": gpu_plan,
                     "budget_bytes": vram_budget, "expert_capacity": vram_experts},
        },
        "warnings": warnings,
    }


def environment_for_plan(plan, env=None, cuda_enabled=True):
    """Apply a plan without overriding explicit user environment settings."""
    result = dict(env or {})
    ram = plan["tiers"]["ram"]
    result.setdefault("RAM_GB", f"{ram['budget_bytes'] / GB:.3f}")

    vram = plan["tiers"]["vram"]
    devices = [device["index"] for device in vram["devices"]]
    if not cuda_enabled or not devices or vram["budget_bytes"] <= 0:
        return result
    if result.get("COLI_CUDA", "1") == "0":
        return result

    result.setdefault("COLI_CUDA", "1")
    if "COLI_GPU" not in result and "COLI_GPUS" not in result:
        key = "COLI_GPU" if len(devices) == 1 else "COLI_GPUS"
        result[key] = ",".join(map(str, devices))
    result.setdefault("CUDA_EXPERT_GB", f"{vram['budget_bytes'] / GB:.3f}")
    if result.get("PIN"):
        result.setdefault("PIN_GB", f"{vram['budget_bytes'] / GB:.3f}")
    return result


def format_bytes(value):
    return f"{value / GB:.1f} GB"


def format_plan(plan):
    model, tiers = plan["model"], plan["tiers"]
    lines = [f"model  {model['shards']} shards · {format_bytes(model['model_bytes'])}",
             f"disk   backing store · {format_bytes(tiers['disk']['available_bytes'])} free",
             f"RAM    {format_bytes(tiers['ram']['budget_bytes'])} budget · "
             f"{format_bytes(tiers['ram']['dense_bytes'])} dense · "
             f"{format_bytes(tiers['ram']['runtime_bytes'])} runtime · "
             f"cap {tiers['ram']['cache_slots_per_layer']}/layer"]
    vram = tiers["vram"]
    if vram["devices"]:
        names = ", ".join(f"{gpu['index']}:{gpu['name']}" for gpu in vram["devices"])
        lines.append(f"VRAM   {format_bytes(vram['budget_bytes'])} hot tier · "
                     f"~{vram['expert_capacity']} experts · {names}")
    else:
        lines.append("VRAM   no NVIDIA device detected · CPU path")
    lines.extend(f"warn   {warning}" for warning in plan["warnings"])
    return "\n".join(lines)
