from __future__ import annotations

import json
from dataclasses import dataclass, field as dc_field
from pathlib import Path

from lockpick_ai.config import ROIConfig, RuntimeConfig


@dataclass(frozen=True)
class FieldConfig:
    left: float = 4.0
    right: float = 631.0
    top: float = 0.0
    top_exit_y: float = 55.0
    paddle_y: float = 730.0
    paddle_contact_y: float = 695.0
    loss_y: float = 830.0
    brick_bottom_y: float = 500.0


@dataclass(frozen=True)
class DisplayConfig:
    auto_scale: bool = True
    auto_content_align: bool = True
    reference_width: int = 1920
    reference_height: int = 1080
    aspect_tolerance: float = 0.03
    window_executable: str = "dota2.exe"
    require_window: bool = True
    window_refresh_seconds: float = 1.0


@dataclass(frozen=True)
class VisionConfig:
    playfield_top: int = 94
    paddle_band_top: int = 710
    paddle_band_bottom: int = 748
    paddle_min_width: int = 70
    aim_band_top: int = 520
    aim_band_bottom: int = 705
    aim_min_points: int = 4
    prompt_left: int = 215
    prompt_top: int = 485
    prompt_right: int = 420
    prompt_bottom: int = 540
    prompt_min_pixels: int = 250
    prompt_confirm_frames: int = 3
    prompt_release_frames: int = 3
    prompt_min_mask_iou: float = 0.65
    rejected_region_radius_px: float = 38.0
    rejected_region_frames: int = 45
    boot_track_max_distance: float = 95.0
    motion_threshold: int = 24
    boot_cyan_h_min: int = 84
    boot_cyan_h_max: int = 108
    boot_search_bottom: int = 710
    boot_arc_min_radius_px: float = 12.0
    boot_arc_max_radius_px: float = 30.0
    boot_arc_max_median_error_px: float = 3.5
    boot_arc_min_score: float = 0.35
    boot_safe_zone_top: float = 500.0
    boot_rebound_track_max_distance: float = 190.0
    boot_motion_confirm_frames: int = 3
    boot_motion_confirm_min_displacement_px: float = 3.0
    launch_prior_frames: int = 24
    launch_prior_tolerance_px: float = 48.0


@dataclass(frozen=True)
class ScoreConfig:
    left: int = 480
    top: int = 40
    right: int = 600
    bottom: int = 72
    stable_frames: int = 3
    min_digit_height: int = 12
    max_delta_per_event: int = 1000


@dataclass(frozen=True)
class TrackingConfig:
    history_ms: float = 180.0
    min_samples: int = 2
    max_jump_px: float = 110.0
    max_speed_px_s: float = 2200.0
    lost_timeout_ms: float = 700.0
    stationary_timeout_ms: float = 450.0
    stationary_radius_px: float = 6.0
    min_frame_displacement_px: float = 3.0
    low_motion_reject_frames: int = 2
    min_valid_speed_px_s: float = 120.0
    no_brick_zone_y: float = 500.0
    max_tracking_y: float = 710.0


@dataclass(frozen=True)
class ControlConfig:
    key_mode: str = "ad"
    paddle_deadzone_px: float = 14.0
    prediction_lead_ms: float = 28.0
    launch_position_ratio: float = 0.50
    launch_target_dx: float = 190.0
    launch_aim_tolerance_px: float = 12.0
    auto_adjust_aim: bool = False
    state_confirm_ms: float = 180.0
    space_cooldown_ms: float = 800.0
    space_retry_ms: float = 2000.0
    space_hold_ms: float = 90.0
    auto_launch: bool = True
    lost_prediction_hold_ms: float = 0.0


@dataclass(frozen=True)
class AppConfig:
    roi: ROIConfig = dc_field(default_factory=lambda: ROIConfig(644, 62, 635, 956))
    display: DisplayConfig = dc_field(default_factory=DisplayConfig)
    field: FieldConfig = dc_field(default_factory=FieldConfig)
    vision: VisionConfig = dc_field(default_factory=VisionConfig)
    score: ScoreConfig = dc_field(default_factory=ScoreConfig)
    tracking: TrackingConfig = dc_field(default_factory=TrackingConfig)
    control: ControlConfig = dc_field(default_factory=ControlConfig)
    runtime: RuntimeConfig = dc_field(default_factory=RuntimeConfig)


def load_config(path: str | Path) -> AppConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return AppConfig(
        roi=ROIConfig(**raw.get("roi", {})),
        display=DisplayConfig(**raw.get("display", {})),
        field=FieldConfig(**raw.get("field", {})),
        vision=VisionConfig(**raw.get("vision", {})),
        score=ScoreConfig(**raw.get("score", {})),
        tracking=TrackingConfig(**raw.get("tracking", {})),
        control=ControlConfig(**raw.get("control", {})),
        runtime=RuntimeConfig(**raw.get("runtime", {})),
    )
