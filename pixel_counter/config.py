from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class BeatCounterConfig:
    sensitivity: float = 1.2

    prom0: float = 0.002
    min_peak_distance_s: float = 0.015
    min_width_s: float = 0.010
    max_width_s: float = 0.80
    min_main_prom: float = 0.012
    min_secondary_prom: float = 0.011

    orient_edge_s: float = 0.25
    orient_min_sep_s: float = 0.08
    orient_n_events: int = 12
    orient_smooth_ms: float = 7.0
    orient_conf_min: float = 0.65
    orientation_sanity_rescue_ratio_penalty: float = 1.00
    orientation_sanity_promoted_fail_penalty: float = 0.35

    drift_window_s: float = 6.0
    drift_quantile: float = 0.20
    drift_apply_min_ratio: float = 0.18
    drift_max_row_corr_for_apply: float = 0.90
    drift_strength: float = 1.0

    vertical_artifact_clean_enabled: bool = True
    vertical_artifact_q99_mult: float = 2.8
    vertical_artifact_q999_mult: float = 0.55
    vertical_artifact_asym_mult: float = 1.15
    vertical_artifact_dedup_gap_samples: int = 2
    vertical_artifact_interp_halfwin: int = 1
    vertical_artifact_sparse_max_count: int = 10
    vertical_artifact_reject_count: int = 11
    vertical_artifact_reject_min_periodicity: float = 0.35
    discontinuity_jump_mad_mult: float = 8.0
    discontinuity_jump_q999_mult: float = 0.9
    discontinuity_bipolar_min_frac: float = 0.45
    discontinuity_peak_exclusion_s: float = 0.020

    main_spike_filter_enabled: bool = True
    main_spike_filter_min_snr: float = 3.0
    main_spike_max_width_s: float = 0.042
    main_spike_max_jump_prom_ratio: float = 0.82
    main_spike_min_abs_jump: float = 0.010
    main_spike_curvature_mad_mult: float = 7.0

    dedup_min_sep_s: float = 0.02
    # Primary close-peak dedup rule: one beat per refractory window.
    refractory_dedup_window_s: float = 0.28
    # Single explicit exception for true close double-lobe morphology.
    main_double_lobe_exception_enabled: bool = True
    main_double_lobe_max_dt_s: float = 0.45
    main_double_lobe_min_valley_ratio: float = 0.14
    main_double_lobe_min_reascent_ratio: float = 0.25
    main_double_lobe_min_weak_width_s: float = 0.070
    main_double_lobe_min_weak_prom_abs: float = 0.030
    main_double_lobe_min_weak_amp_rel_strong: float = 0.65
    main_shadow_max_dt_s: float = 0.14
    main_shadow_narrow_width_s: float = 0.03
    main_shadow_next_min_width_s: float = 0.06
    main_shadow_next_prom_ratio: float = 1.05
    main_duplicate_max_dt_s: float = 0.10
    main_post_pseudo_max_dt_s: float = 0.18
    main_post_pseudo_rel_prom_max: float = 0.62
    main_post_pseudo_rel_width_max: float = 0.62
    main_post_pseudo_rel_amp_max: float = 0.92
    main_plateau_dedup_max_dt_s: float = 0.18
    main_plateau_dedup_min_width_s: float = 0.18
    main_plateau_dedup_rel_amp_diff_max: float = 0.03
    main_plateau_dedup_rel_prom_diff_max: float = 0.05
    main_plateau_dedup_rel_width_diff_max: float = 0.08
    main_close_pair_max_dt_s: float = 0.16
    main_close_pair_replace_score_ratio: float = 1.08
    main_close_pair_replace_min_width_ratio: float = 0.82
    main_same_transient_weak_rel_prom_max: float = 0.62
    main_same_transient_weak_rel_amp_max: float = 0.65
    main_same_transient_weak_max_width_s: float = 0.06
    main_local_weak_filter_enabled: bool = True
    main_local_weak_rel_prom_max: float = 0.28
    main_local_weak_rel_amp_max: float = 0.55
    main_local_weak_rel_width_max: float = 0.70
    main_local_weak_min_gap_rel_ibi: float = 0.58
    main_local_tiny_filter_enabled: bool = True
    main_local_tiny_rel_prom_max: float = 0.60
    main_local_tiny_rel_amp_max: float = 0.62
    main_local_tiny_rel_width_max: float = 0.55
    main_short_gap_filter_enabled: bool = True
    main_short_gap_rel_ibi: float = 0.58
    main_short_gap_floor_s: float = 0.32
    main_short_gap_weak_prom_ratio: float = 0.48
    main_short_gap_weak_width_ratio: float = 0.68
    main_short_gap_weak_amp_ratio: float = 0.90
    max_main_per_transient_ratio: float = 5.1
    main_interbeat_tiny_filter_enabled: bool = True
    main_interbeat_tiny_min_gap_s: float = 0.30
    main_interbeat_tiny_rel_amp_max: float = 0.42
    main_interbeat_tiny_rel_prom_max: float = 0.46
    main_interbeat_tiny_max_width_s: float = 0.14
    main_interbeat_tiny_global_amp_rel_max: float = 0.55
    main_interbeat_tiny_global_prom_rel_max: float = 0.60

    qc_min_rows: int = 3
    # Consolidated mixed-orientation QC rule.
    # Rejects when a weighted combination of minor-polarity fraction, row-direction
    # transition, low periodicity, and low row-correlation indicates inconsistent morphology.
    qc_mixed_orientation_minor_min: float = 0.22
    qc_mixed_orientation_transition_min: float = 0.30
    qc_mixed_orientation_periodicity_max: float = 0.58
    qc_mixed_orientation_corr_max: float = 0.75
    qc_mixed_orientation_weight_minor: float = 0.35
    qc_mixed_orientation_weight_transition: float = 0.30
    qc_mixed_orientation_weight_periodicity: float = 0.20
    qc_mixed_orientation_weight_corr: float = 0.15
    qc_mixed_orientation_flip_bonus: float = 0.20
    qc_mixed_orientation_reject_score: float = 0.65
    qc_mixed_orientation_gate_minor: float = 0.18
    qc_mixed_orientation_gate_transition: float = 0.20
    qc_mixed_flip_transition_min: float = 0.50
    qc_mixed_flip_spread_min: float = 0.16
    qc_mixed_flip_rel_strength_min: float = 0.45
    qc_low_snr_threshold: float = 2.6
    qc_low_snr_periodicity_min: float = 0.18
    qc_nonperiodic_spiky_max_periodicity: float = 0.15
    qc_nonperiodic_spiky_min_spike_fraction: float = 0.05
    qc_nonstationary_min_rows: int = 8
    qc_nonstationary_min_early_strength: float = 4.0
    qc_nonstationary_drop_ratio: float = 0.45
    qc_nonstationary_min_late_strength: float = 1.8
    qc_nonstationary_max_late_periodicity: float = 0.22
    qc_nonstationary_coherent_periodicity_min: float = 0.60
    qc_nonstationary_noise_periodicity_max: float = 0.12
    qc_nonstationary_noise_snr_max: float = 2.4
    qc_nonstationary_noise_spike_min: float = 0.025

    use_narrow_retry: bool = True

    main_gap_fill_enabled: bool = True
    main_gap_fill_periodicity_min: float = 0.75
    main_gap_fill_corr_min: float = 0.70
    main_gap_fill_gap_mult: float = 1.7
    main_gap_fill_candidate_rel_strong: float = 0.30
    main_gap_fill_candidate_min_width_s: float = 0.040
    main_gap_fill_min_sep_s: float = 0.12
    main_gap_fill_max_add_per_gap: int = 1
    main_transient_fill_enabled: bool = True
    main_transient_fill_periodicity_min: float = 0.72
    main_transient_fill_corr_min: float = 0.68
    main_transient_fill_min_prom_rel_strong: float = 0.16
    main_transient_fill_min_amp_rel_ref: float = 0.40
    main_transient_fill_min_sep_s: float = 0.10
    main_transient_replace_amp_ratio: float = 1.35
    main_transient_replace_main_prom_rel_strong_max: float = 0.70
    main_transient_replace_score_ratio: float = 1.16
    main_transient_tail_n: int = 3
    main_transient_tail_min_amp_rel_ref: float = 0.40
    main_transient_edge_window_s: float = 0.06
    main_transient_edge_amp_rel_ref: float = 0.95

    rescue_enabled: bool = True
    rescue_window_s: float = 0.22
    rescue_gap_mult: float = 1.55
    rescue_min_gap_s: float = 0.55
    rescue_min_sep_s: float = 0.08
    rescue_min_width_s: float = 0.014
    rescue_max_width_s: float = 0.75
    rescue_min_sep_rel_ibi: float = 0.16
    rescue_min_prom_rel_ref: float = 0.30
    rescue_min_amp_rel_ref: float = 0.40
    rescue_boundary_min_prom_rel_ref: float = 0.12
    rescue_boundary_min_width_s: float = 0.070
    rescue_boundary_amp_rel_ref: float = 0.86
    rescue_min_rel_amp_strict: float = 0.52
    rescue_min_rel_prom_strict: float = 0.50
    rescue_min_rel_width_strict: float = 0.45
    rescue_min_neighbor_gap_rel_ibi: float = 0.18
    rescue_prom_quantile: float = 0.22
    rescue_max_candidates_per_window: int = 1
    rescue_overlap_min_main_width_s: float = 0.16
    rescue_overlap_width_factor: float = 1.05
    rescue_replace_prev_enabled: bool = True
    rescue_replace_prev_rel_prom: float = 0.45
    rescue_replace_prev_rel_width: float = 0.75
    rescue_replace_prev_rel_amp: float = 0.55
    rescue_replace_prev_max_dt_rel_ibi: float = 0.90
    rescue_replace_prev_max_dt_s: float = 0.90

    show_threshold_guides: bool = True


@dataclass(frozen=True)
class AFCReviewConfig:
    enabled: bool = False
    review_all_main_peaks: bool = True
    only_review_windows_with_candidates: bool = False
    window_start_delay_s: float = 0.04
    window_end_delay_s: float = 0.55
    lower_line_mode: str = "offset_from_main"
    upper_line_mode: str = "offset_from_main"
    default_lower_line_offset: float = -0.28
    default_upper_line_offset: float = -0.02
    default_afc_lower_left_value: float = float("nan")
    default_afc_lower_right_value: float = float("nan")
    default_afc_upper_left_value: float = float("nan")
    default_afc_upper_right_value: float = float("nan")
    # Backward-compatible aliases (legacy names).
    default_afc_left_value: float = float("nan")
    default_afc_right_value: float = float("nan")
    default_afc_upper_cap_value: float = float("nan")
    default_afc_upper_cap_quantile: float = 0.95
    default_afc_upper_cap_fallback: float = 1.0
    prefer_report_upper_line: bool = True
    default_afc_left_quantile: float = 0.68
    default_afc_right_quantile: float = 0.68
    default_afc_quantile_floor: float = 0.35
    default_afc_quantile_ceil: float = 0.92
    min_secondary_prominence: float = 0.010
    min_secondary_width_s: float = 0.010
    max_secondary_width_s: float = 0.400
    min_secondary_distance_s: float = 0.050
    main_peak_exclusion_window_s: float = 0.060
    save_review_json: bool = True
    save_review_csv: bool = True
    save_review_png: bool = True
    save_partial_progress: bool = True
    resume_existing_session: bool = False
    interactive_backend: str = "matplotlib"
    review_output_suffix: str = "_afc_review"
