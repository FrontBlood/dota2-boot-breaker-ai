from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ROIConfig:
    left: int = 1000
    top: int = 250
    width: int = 620
    height: int = 620


@dataclass(frozen=True)
class DialConfig:
    center_x: float = 310.0
    center_y: float = 310.0
    radius: float = 270.0
    angle_bins: int = 720
    pointer_inner_ratio: float = 0.12
    pointer_outer_ratio: float = 0.86
    zone_inner_ratio: float = 0.72
    zone_outer_ratio: float = 0.94


@dataclass(frozen=True)
class HSVRange:
    h_min: int
    h_max: int
    s_min: int
    s_max: int
    v_min: int
    v_max: int


@dataclass(frozen=True)
class VisionConfig:
    yellow_hsv: HSVRange = field(default_factory=lambda: HSVRange(15, 42, 90, 255, 100, 255))
    blue_hsv: HSVRange = field(default_factory=lambda: HSVRange(85, 130, 70, 255, 80, 255))
    pointer_hsv: HSVRange = field(default_factory=lambda: HSVRange(0, 179, 0, 105, 150, 255))
    pointer_min_score: float = 0.34
    zone_pixel_ratio: float = 0.35
    zone_min_width_deg: float = 3.0
    smooth_bins: int = 5
    adaptive_zone_color: bool = True
    zone_min_value: int = 10
    zone_min_saturation: int = 22
    yellow_opponent_min: float = 0.10
    blue_opponent_min: float = 0.055
    blue_zone_max_value: int = 175


@dataclass(frozen=True)
class TrackingConfig:
    history_ms: float = 220.0
    min_samples: int = 4
    max_jump_deg: float = 65.0
    max_abs_speed_dps: float = 1800.0


@dataclass(frozen=True)
class ControlConfig:
    input_latency_ms: float = 52.0
    latency_jitter_ms: float = 10.0
    base_margin_deg: float = 2.5
    speed_error_fraction: float = 0.05
    max_safe_speed_dps: float = 850.0
    click_cooldown_ms: float = 140.0
    feedback_timeout_ms: float = 350.0
    rmb_accel_enabled: bool = True
    rmb_release_distance_deg: float = 28.0
    rmb_max_accel_speed_dps: float = 650.0
    blue_time_value_points: float = 120.0


@dataclass(frozen=True)
class RuntimeConfig:
    show_debug: bool = True
    record_telemetry: bool = True
    idle_sleep_ms: int = 1
    debug_scale: float = 0.9
    record_score_samples: bool = False
    debug_topmost: bool = True
    debug_auto_position: bool = True


@dataclass(frozen=True)
class AppConfig:
    roi: ROIConfig = field(default_factory=ROIConfig)
    dial: DialConfig = field(default_factory=DialConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def _construct(cls: type, values: dict[str, Any]):
    return cls(**values)


def load_config(path: str | Path) -> AppConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    vision_raw = dict(raw.get("vision", {}))
    for key in ("yellow_hsv", "blue_hsv", "pointer_hsv"):
        if key in vision_raw:
            vision_raw[key] = _construct(HSVRange, vision_raw[key])
    return AppConfig(
        roi=_construct(ROIConfig, raw.get("roi", {})),
        dial=_construct(DialConfig, raw.get("dial", {})),
        vision=_construct(VisionConfig, vision_raw),
        tracking=_construct(TrackingConfig, raw.get("tracking", {})),
        control=_construct(ControlConfig, raw.get("control", {})),
        runtime=_construct(RuntimeConfig, raw.get("runtime", {})),
    )
