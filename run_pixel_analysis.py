import argparse
import sys
from pathlib import Path

from pixel_counter.io_utils import build_output_path

from pixel_counter import (
    AFCReviewConfig,
    BeatCounterConfig,
    analyze_raw_cytocypher_workbook,
    analyze_workbook_with_afc_review,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PixelCorrelation analysis on one workbook.")
    parser.add_argument("--input", required=True, help="Path to input .xlsx workbook")
    parser.add_argument(
        "--afc-review",
        action="store_true",
        help="Enable AFC review mode (non-interactive by default unless --afc-interactive is set)",
    )
    parser.add_argument(
        "--afc-interactive",
        action="store_true",
        help="Enable interactive AFC GUI (requires --afc-review)",
    )
    parser.add_argument(
        "--afc-resume",
        action="store_true",
        help="Resume existing AFC session if present (used only with --afc-review)",
    )
    parser.add_argument(
        "--debug-peak-trace",
        action="store_true",
        help="Write peak debug exports",
    )
    args = parser.parse_args()

    if args.afc_interactive and not args.afc_review:
        parser.error("--afc-interactive requires --afc-review")
    if args.afc_resume and not args.afc_review:
        parser.error("--afc-resume requires --afc-review")

    raw_path = str(Path(args.input).expanduser().resolve())
    stim_hz = 1.0
    recording_s = 10.0

    report_docx = build_output_path(raw_path, "_arrhythmia_report.docx")
    summary_xlsx = build_output_path(raw_path, "_arrhythmia_summary.xlsx")
    diagnostics_dir = build_output_path(raw_path, "_diagnostics").replace(" ", "_")

    auto_config = BeatCounterConfig(sensitivity=1.2)
    afc_config = AFCReviewConfig(
        enabled=bool(args.afc_review),
        resume_existing_session=bool(args.afc_resume),
    )

    if args.afc_review:
        summary_df = analyze_workbook_with_afc_review(
            raw_xlsx_path=raw_path,
            stim_hz=stim_hz,
            recording_s=recording_s,
            config=auto_config,
            afc_config=afc_config,
            output_docx=report_docx,
            output_summary_xlsx=summary_xlsx,
            diagnostics_dir=diagnostics_dir,
            debug=False,
            debug_peak_trace=bool(args.debug_peak_trace),
            show_plots=False,
            interactive_review=bool(args.afc_interactive),
        )
    else:
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

    if args.afc_review:
        print("\nAFC session JSON:")
        print(build_output_path(raw_path, "_afc_review_session.json"))
        print("\nAFC events CSV:")
        print(build_output_path(raw_path, "_afc_events.csv"))
        print("\nAFC review log CSV:")
        print(build_output_path(raw_path, "_afc_review_log.csv"))
        print("\nAFC review plots dir:")
        print(build_output_path(raw_path, "_afc_review_plots"))
    if args.debug_peak_trace:
        print("\nPeak debug XLSX:")
        print(build_output_path(raw_path, "_peak_debug.xlsx"))
        print("\nPeak debug CSV:")
        print(build_output_path(raw_path, "_peak_debug.csv"))

    if "Status" in summary_df.columns and (summary_df["Status"] == "ERROR").any():
        sys.exit(1)


if __name__ == "__main__":
    main()
