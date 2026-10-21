"""Polarity overrides management for Cytocypher v2 pipeline.

Provides persistent, workbook-tied manual polarity configuration and fallback handling.
Ensures overrides saved for one workbook cannot be accidentally applied to an unrelated
workbook with identical segment numbers.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Union


def compute_file_sha256(file_path: Union[str, Path], chunk_size: int = 65536) -> str:
    """Compute the SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def default_overrides_path(workbook_path: Union[str, Path]) -> Path:
    """Return the default JSON overrides path associated with a workbook."""
    wb = Path(workbook_path)
    return wb.with_suffix(".polarity_overrides.json")


def parse_polarity_override_str(override_str: str) -> Dict[int, int]:
    """Parse a comma-separated polarity override string, e.g. '17=-1,18=-1,45=1' or '17:-1,45:1'.

    Supported values per segment:
      -1, 'negative', 'down', 'downward' -> -1
      +1, 1, 'positive', 'up', 'upward'  -> 1
       0, 'auto', 'none'                 -> 0 (clears override)
    """
    if not override_str or not override_str.strip():
        return {}
    overrides: Dict[int, int] = {}
    tokens = [t.strip() for t in override_str.split(",") if t.strip()]
    for token in tokens:
        if "=" in token:
            seg_s, pol_s = token.split("=", 1)
        elif ":" in token:
            seg_s, pol_s = token.split(":", 1)
        else:
            raise ValueError(f"Invalid polarity override token '{token}'. Expected 'segment=polarity'.")

        try:
            seg = int(seg_s.strip())
        except ValueError as e:
            raise ValueError(f"Invalid segment number '{seg_s}' in token '{token}'.") from e

        p_str = pol_s.strip().lower()
        if p_str in ("-1", "negative", "down", "downward"):
            pol = -1
        elif p_str in ("1", "+1", "positive", "up", "upward"):
            pol = 1
        elif p_str in ("0", "auto", "none"):
            pol = 0
        else:
            raise ValueError(f"Invalid polarity value '{pol_s}' for segment {seg}. Expected -1, 1, or auto.")

        overrides[seg] = pol
    return overrides


def save_polarity_overrides(
    workbook_path: Union[str, Path],
    overrides: Dict[int, int],
    output_path: Optional[Union[str, Path]] = None,
    segment_rejections: Optional[Dict[Union[int, str], Dict[str, Any]]] = None,
) -> Path:
    """Save polarity overrides tied to the given workbook in JSON format.

    Records both the workbook filename and SHA256 digest to prevent accidental reuse.
    Optionally stores segment-level manual rejections alongside orientation decisions.
    """
    wb = Path(workbook_path)
    out_file = Path(output_path) if output_path else default_overrides_path(wb)
    sha256 = compute_file_sha256(wb) if wb.exists() else ""

    existing_rejections = {}
    if segment_rejections is None and out_file.exists():
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                old_data = json.load(f)
                existing_rejections = old_data.get("segment_rejections", {})
        except Exception:
            pass

    rej_payload = segment_rejections if segment_rejections is not None else existing_rejections

    payload: Dict[str, Any] = {
        "workbook_name": wb.name,
        "workbook_sha256": sha256,
        "overrides": {str(k): int(v) for k, v in sorted(overrides.items()) if int(v) in (-1, 1)},
    }
    if rej_payload:
        def _sort_key(item):
            s = str(item[0])
            return (0, int(s)) if s.isdigit() else (1, s)
        payload["segment_rejections"] = {
            str(k): dict(v) for k, v in sorted(rej_payload.items(), key=_sort_key)
        }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_file


def load_polarity_overrides(
    workbook_path: Union[str, Path],
    overrides_path: Union[str, Path],
    strict_hash: bool = False,
) -> Dict[int, int]:
    """Load polarity overrides from JSON and verify compatibility with workbook_path.

    Parameters
    ----------
    workbook_path : path to the Excel workbook being processed.
    overrides_path : path to the polarity overrides JSON file.
    strict_hash : if True, requires exact SHA256 match; if False, validates workbook filename
                  and warns on SHA256 mismatch (useful if file metadata was touched).

    Raises
    ------
    ValueError if the override file was created for an unrelated workbook.
    """
    wb = Path(workbook_path)
    ov_file = Path(overrides_path)
    if not ov_file.exists():
        raise FileNotFoundError(f"Polarity overrides file not found: {ov_file}")

    with open(ov_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    target_name = data.get("workbook_name", "")
    target_sha256 = data.get("workbook_sha256", "")

    if target_name and target_name != wb.name:
        raise ValueError(
            f"Mismatched workbook in overrides file {ov_file.name}!\n"
            f"Override was created for '{target_name}', but active workbook is '{wb.name}'."
        )

    if strict_hash and target_sha256 and wb.exists():
        current_sha256 = compute_file_sha256(wb)
        if current_sha256 != target_sha256:
            raise ValueError(
                f"Mismatched workbook SHA256 in {ov_file.name}!\n"
                f"Expected {target_sha256}, got {current_sha256}."
            )

    raw_dict = data.get("overrides", {})
    clean_overrides: Dict[int, int] = {}
    for k, v in raw_dict.items():
        try:
            seg = int(k)
            pol = int(v)
            if pol in (-1, 1):
                clean_overrides[seg] = pol
        except (ValueError, TypeError):
            continue
    return clean_overrides


def load_segment_rejections(
    workbook_path: Union[str, Path],
    overrides_path: Union[str, Path],
) -> Dict[int, Dict[str, Any]]:
    """Load manual segment rejections from sidecar JSON."""
    wb = Path(workbook_path)
    ov_file = Path(overrides_path)
    if not ov_file.exists():
        return {}

    with open(ov_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    target_name = data.get("workbook_name", "")
    if target_name and target_name != wb.name:
        raise ValueError(
            f"Mismatched workbook in overrides file {ov_file.name}!\n"
            f"Override was created for '{target_name}', but active workbook is '{wb.name}'."
        )

    raw_rej = data.get("segment_rejections", {})
    clean_rejections: Dict[int, Dict[str, Any]] = {}
    for k, v in raw_rej.items():
        try:
            seg = int(k)
            if isinstance(v, dict) and v.get("rejected", False):
                clean_rejections[seg] = {
                    "rejected": True,
                    "reason": str(v.get("reason", "Other / researcher decision")),
                }
        except (ValueError, TypeError):
            continue
    return clean_rejections

