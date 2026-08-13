"""Presentation helpers for the local Streamlit runner."""

from pathlib import Path


SUMMARY_SEPARATOR = "=" * 100
EXECUTION_TIME_PREFIX = "Execution time in minutes:"


def separate_execution_time(output: str) -> tuple[str, str]:
    """Separate execution-time metadata from ordinary streamed output."""
    retained_lines = []
    execution_time = ""
    for line in output.splitlines(keepends=True):
        if line.strip().startswith(EXECUTION_TIME_PREFIX):
            execution_time = line.strip()
        else:
            retained_lines.append(line)
    return "".join(retained_lines), execution_time


def extract_evaluation_summary(output: str) -> str:
    """Extract compact aggregate metrics from verbose evaluation output."""
    summary_prefixes = (
        "Evaluation Summary -",
        "orig_avg_abs_count_diff:",
        "orig_avg_abs_TRL_diff:",
        "orig_avg_abs_dia_diff:",
        "color_correct:",
        "color_macro_precision:",
        "Avg of per image rel_error of TRL:",
        "Avg of per image rel_error of diameter:",
        "Avg of per image absolute error of color:",
    )
    lines = []
    for line in output.splitlines():
        stripped_line = line.strip()
        if stripped_line == "Evaluation Summary - keypoint detection":
            continue
        if stripped_line.startswith(summary_prefixes):
            lines.append(stripped_line)
    return "\n".join(lines)


def append_evaluation_summary(text_results_path: str) -> str:
    """Append the GUI's consolidated evaluation summary to a text artifact."""
    results_path = Path(text_results_path)
    if not results_path.is_file():
        return ""
    output = results_path.read_text(encoding="utf-8")
    summary = extract_evaluation_summary(output)
    if not summary:
        return ""
    section = (
        f"\n{SUMMARY_SEPARATOR}\n"
        "Evaluation Summary\n"
        f"{SUMMARY_SEPARATOR}\n"
        f"{summary}\n"
    )
    with results_path.open("a", encoding="utf-8") as results_file:
        results_file.write(section)
    return section
