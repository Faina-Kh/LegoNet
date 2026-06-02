from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import streamlit as st


DATASET_OPTIONS = ("roots", "grapes")
NETWORK_OPTIONS = (
    "bbox_detection",
    "counting_lean",
    "counting_reg",
    "both",
    "both_for_roots_2",
    "both_Back2bFind2b",
)
RUN_MODES = ("Inference", "Training")
VAL_SETS = ("Test", "Val")
ESTIMATE_TYPES = ("withKeyPoints", "reg_fpn_p3_p7_min_sig")


def find_project_root(start: Path) -> Path:
    """Find the repository root from this Streamlit app file."""
    for path in (start, *start.parents):
        if (path / "legonet" / "scripts" / "main.py").exists():
            return path
    raise FileNotFoundError("Could not find legonet/scripts/main.py above the app directory.")


def bool_arg(value: bool) -> str:
    """Convert a checkbox value to the string expected by main.py."""
    return "true" if value else "false"


def build_command(
    main_script: Path,
    storage_path: str,
    dataset_name: str,
    network_type: str,
    run_script: str,
    val_set: str,
    gpu_num: str,
    current_results_dir: str,
    estimate_type: str,
    num_of_epochs: int,
    have_gt: bool,
    to_draw: bool,
    evaluate_detection: bool,
    load_weights: bool,
) -> list[str]:
    """Build the LegoNet main.py command from GUI settings."""
    return [
        sys.executable,
        str(main_script),
        "--storage-path",
        storage_path,
        "--dataset-name",
        dataset_name,
        "--network-type",
        network_type,
        "--run-script",
        run_script,
        "--val-set",
        val_set,
        "--gpu-num",
        gpu_num,
        "--current-results-dir",
        current_results_dir,
        "--estimate-type",
        estimate_type,
        "--num-of-epochs",
        str(num_of_epochs),
        "--have-gt",
        bool_arg(have_gt),
        "--to-draw",
        bool_arg(to_draw),
        "--load-weights",
        bool_arg(load_weights),
    ]


def format_command(command: list[str]) -> str:
    """Format a command for display in the UI."""
    return " ".join(f'"{part}"' if " " in part else part for part in command)


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
MAIN_SCRIPT = PROJECT_ROOT / "legonet" / "scripts" / "main.py"

st.set_page_config(page_title="LegoNet Runner", page_icon="L", layout="wide")

st.title("LegoNet Runner")

with st.sidebar:
    st.header("Run Settings")
    storage_path = st.text_input(
        "Storage path",
        value=r"C:\Users\bordezki\Desktop\LegoNet",
    )
    dataset_name = st.selectbox("Dataset", DATASET_OPTIONS, index=0)
    network_type = st.selectbox("Network type", NETWORK_OPTIONS, index=0)
    run_script = st.selectbox("Run mode", RUN_MODES, index=0)
    val_set = st.selectbox("Validation set", VAL_SETS, index=0)
    gpu_num = st.text_input("GPU number", value="0")

    st.header("Advanced")
    current_results_dir = st.text_input("Current results dir", value="bbox_detection")
    estimate_type = st.selectbox("Estimate type", ESTIMATE_TYPES, index=0)
    num_of_epochs = st.number_input("Number of epochs", min_value=1, value=2, step=1)
    have_gt = st.checkbox("Have ground truth", value=True)
    to_draw = st.checkbox("Draw visualizations", value=False)
    load_weights = st.checkbox("Load weights", value=True)

command = build_command(
    main_script=MAIN_SCRIPT,
    storage_path=storage_path,
    dataset_name=dataset_name,
    network_type=network_type,
    run_script=run_script,
    val_set=val_set,
    gpu_num=gpu_num,
    current_results_dir=current_results_dir,
    estimate_type=estimate_type,
    num_of_epochs=int(num_of_epochs),
    have_gt=have_gt,
    to_draw=to_draw,
    load_weights=load_weights,
)

left, right = st.columns([2, 1])

with left:
    st.subheader("Command Preview")
    st.code(format_command(command), language="powershell")

with right:
    st.subheader("Project")
    st.text_input("Project root", value=str(PROJECT_ROOT), disabled=True)
    st.text_input("Python", value=sys.executable, disabled=True)

run_clicked = st.button("Run LegoNet", type="primary")

if run_clicked:
    if not storage_path.strip():
        st.error("Storage path is required.")
        st.stop()

    with st.spinner("Running LegoNet..."):
        result = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )

    if result.returncode == 0:
        st.success("Run completed successfully.")
    else:
        st.error(f"Run failed with exit code {result.returncode}.")

    if result.stdout:
        st.subheader("Output")
        st.code(result.stdout)

    if result.stderr:
        st.subheader("Errors")
        st.code(result.stderr)
