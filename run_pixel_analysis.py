import argparse
import sys
from pathlib import Path

from pixel_counter.io_utils import build_output_path

from pixel_counter import (
    BeatCounterConfig,
    analyze_raw_cytocypher_workbook,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run contraction frequency analysis on one workbook.")
    parser.add_argument("--input", required=True, help="Path to input .xlsx workbook")
    parser.add_argument(
        "--afc-review",
        action="store_true",
        help="[REMOVED] AFC review workflow has been removed.",
    )
    parser.add_argument(
        "--afc-interactive",
        action="store_true",
        help="[REMOVED] AFC review workflow has been removed.",
    )
    parser.add_argument(
        "--afc-resume",
        action="store_true",
        help="[REMOVED] AFC review workflow has been removed.",
    )
    parser.add_argument(
        "--debug-peak-trace",
        action="store_true",
        help="Write peak debug exports",
    )
    args = parser.parse_args()

    if args.afc_review or args.afc_interactive or args.afc_resume:
        parser.error("AFC review workflow has been removed. Please remove AFC flags from your command.")

    raw_path = str(Path(args.input).expanduser().resolve())
    stim_hz = 1.0
    recording_s = 10.0

    report_docx = build_output_path(raw_path, "_contraction_report.docx")
    summary_xlsx = build_output_path(raw_path, "_contraction_summary.xlsx")
    diagnostics_dir = build_output_path(raw_path, "_diagnostics").replace(" ", "_")

    auto_config = BeatCounterConfig(sensitivity=1.2)

    summary_df = analyze_raw_cytocypher_workbook(
        raw_xlsx_path=raw_path,
        stim_hz=stim_hz,
        recording_s=recording_s,
        config=auto_config,
        output_docx=report_docx,
        output_summary_xlsx=summary_xlsx,
        diagnostics_dir=diagnostics_dir,
        debug=True,
        debug_peak_trace=True,
        show_plots=False,
    )

    print(summary_df.to_string(index=False))
    print("\nDOCX report saved to:")
    print(report_docx)
    print("\nSummary XLSX saved to:")
    print(summary_xlsx)

    if args.debug_peak_trace:
        print("\nPeak debug XLSX:")
        print(build_output_path(raw_path, "_peak_debug.xlsx"))
        print("\nPeak debug CSV:")
        print(build_output_path(raw_path, "_peak_debug.csv"))

    if "Status" in summary_df.columns and (summary_df["Status"] == "ERROR").any():
        sys.exit(1)


if __name__ == "__main__":
    main()
