"""Presentation helpers for the local Streamlit runner."""


def extract_evaluation_summary(output: str) -> str:
    """Extract compact aggregate metrics from verbose evaluation output."""
    summary_prefixes = (
        "orig_avg_abs_TRL_diff:",
        "orig_avg_abs_dia_diff:",
        "color_correct:",
        "color_macro_precision:",
        "Avg of per image rel_error of TRL:",
        "Avg of per image rel_error of diameter:",
        "Avg of per image absolute error of color:",
        "Points detection evaluation:",
        "mAP:",
    )
    lines = [
        line.strip()
        for line in output.splitlines()
        if line.strip().startswith(summary_prefixes)
    ]
    return "\n".join(lines)
