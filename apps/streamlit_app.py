from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import streamlit as st


DATASET_OPTIONS = ("roots", "grapes")
NETWORK_OPTIONS = (
    "bbox_detection",
    "per_image_estimation_keypoints",
    "per_image_estimation_regression",
    "per_object_counting",
    "per_object_attributes",
    "per_object_attributes_multibranch",
)
NETWORKS_OPTIONS_BY_DATASETS = {'roots': ("bbox_detection", "per_image_estimation_keypoints", "per_image_estimation_regression", "per_object_attributes",
                                          "per_object_attributes_multibranch"),
                                'grapes': ("bbox_detection", "per_object_counting")
                                }
RUN_MODES = ("Inference", "Training")
VAL_SETS = ("Test", "Val")
ESTIMATE_TYPES = ("withKeyPoints", "reg_fpn_p3_p7_min_sig")
OPTIONAL_DETECTION_EVAL_NETWORK_OPTIONS = ("per_object_counting", "per_object_attributes", "per_object_attributes_multibranch")
MANDATORY_DETECTION_EVAL_NETWORK_OPTIONS = ("bbox_detection")
ESTIMATE_SELECT_NETWORK_OPTIONS = ("per_object_counting", "per_object_attributes")
DEFAULT_ESTIMATE_TYPE_BY_NETWORK = {
    "per_image_estimation_keypoints": "withKeyPoints",
    "per_image_estimation_regression": "reg_fpn_p3_p7_min_sig",
    "per_object_attributes_multibranch": "withKeyPoints"
}
WEIGHTS_TYPES = ('full_model_weights', 'partial_weights')


def find_project_root(start: Path) -> Path:
    """Find the repository root from this Streamlit app file."""
    for path in (start, *start.parents):
        if (path / "scripts" / "run_legonet.py").exists():
            return path
    raise FileNotFoundError("Could not find scripts/run_legonet.py above the app directory.")


def bool_arg(value: bool) -> str:
    """Convert a checkbox value to the string expected by main.py."""
    return "true" if value else "false"


def has_cuda_gpu() -> bool:
    """Return whether the active Python environment can see a CUDA GPU."""
    try:
        import torch
    except ImportError:
        return False

    return torch.cuda.is_available()


def choose_local_directory(initial_path: str = "") -> str:
    """Open a native directory picker on the machine running Streamlit."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as error:
        raise RuntimeError("The native folder picker requires Tkinter.") from error

    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except tk.TclError:
            pass

        initial_directory = Path(initial_path).expanduser()
        dialog_options = {"title": "Choose the LegoNet storage directory"}
        if initial_directory.is_dir():
            dialog_options["initialdir"] = str(initial_directory)
        return filedialog.askdirectory(**dialog_options)
    except Exception as error:
        raise RuntimeError(
            "The native folder picker is unavailable. Enter the path manually."
        ) from error
    finally:
        if root is not None:
            root.destroy()


def browse_for_storage_path() -> None:
    """Update Streamlit state from the optional local folder picker."""
    try:
        selected_path = choose_local_directory(
            st.session_state.get("storage_path", "")
        )
    except RuntimeError as error:
        st.session_state.storage_path_dialog_error = str(error)
        return

    if selected_path:
        st.session_state.storage_path = selected_path
    st.session_state.storage_path_dialog_error = ""


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
    weights_type: str,
) -> list[str]:
    """Build the LegoNet main.py command from GUI settings."""
    return [
        sys.executable,
        "-u",
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
        "--evaluate-detection",
        bool_arg(evaluate_detection),
        "--load-weights",
        bool_arg(load_weights),
        "--weights-type",
        str(weights_type),
    ]


def format_command(command: list[str]) -> str:
    """Format a command for display in the UI."""
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def expected_experiment_root(storage_path: str, dataset_name: str, current_results_dir: str) -> Path:
    """Return the experiment folder where main.py writes results for these settings."""
    return Path(storage_path) / "ExpResults" / dataset_name / current_results_dir


def find_recent_artifacts(experiment_root: Path, limit: int = 12) -> list[Path]:
    """Find recent text and CSV outputs below the experiment folder."""
    if not experiment_root.exists():
        return []

    artifacts = [
        path
        for pattern in ("*.csv", "*.txt")
        for path in experiment_root.rglob(pattern)
        if path.is_file()
    ]
    return sorted(artifacts, key=lambda path: path.stat().st_mtime, reverse=True)[:limit]


def read_preview(path: Path, max_chars: int = 4000) -> str:
    """Read a small preview from a result file."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:max_chars]
    except OSError as exc:
        return f"Could not read {path}: {exc}"


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
MAIN_SCRIPT = PROJECT_ROOT / "scripts" / "run_legonet.py"
GPU_AVAILABLE = has_cuda_gpu()

st.set_page_config(page_title="LegoNet Runner", page_icon="L", layout="wide")

st.title("LegoNet Runner")

with st.sidebar:
    st.header("Run Settings")
    if "storage_path" not in st.session_state:
        st.session_state.storage_path = os.environ.get(
            "LEGONET_STORAGE_PATH",
            "",
        )

    storage_path = st.text_input(
        "Storage path",
        key="storage_path",
        help=(
            "Uses LEGONET_STORAGE_PATH as its initial value. The local folder "
            "picker is only available when Streamlit runs on this computer."
        ),
    )
    st.button(
        "Browse local machine…",
        on_click=browse_for_storage_path,
        use_container_width=True,
    )
    folder_dialog_error = st.session_state.get("storage_path_dialog_error", "")
    if folder_dialog_error:
        st.warning(folder_dialog_error)

    dataset_name = st.selectbox("Dataset", DATASET_OPTIONS, index=0)

    network_type = st.selectbox("Network type", NETWORKS_OPTIONS_BY_DATASETS[dataset_name], index=0)

    run_script = st.selectbox("Run mode", RUN_MODES, index=0)

    if run_script == 'Inference':
        val_set = st.selectbox("Validation set", VAL_SETS, index=0)
    else:
        val_set = "Val"

    use_gpu = st.checkbox("Use GPU", value=GPU_AVAILABLE)
    if use_gpu:
        gpu_num = st.text_input("GPU number", value="0")
    else:
        gpu_num = ""

    st.header("Advanced")
    if network_type in ESTIMATE_SELECT_NETWORK_OPTIONS:
        estimate_type = st.selectbox("Estimate type", ESTIMATE_TYPES, index=0)
    else:
        estimate_type = DEFAULT_ESTIMATE_TYPE_BY_NETWORK.get(network_type, "reg_fpn_p3_p7_min_sig")

    if run_script == "Training":
        num_of_epochs = st.number_input("Number of epochs", min_value=1, value=300, step=1)
    else:
        num_of_epochs = 0

    current_results_dir = st.text_input("Current results dir", value=network_type+'_'+val_set+"_Results")

    have_gt = st.checkbox("Have ground truth", value=True)
    to_draw = st.checkbox("Draw visualizations", value=False)
    if network_type in OPTIONAL_DETECTION_EVAL_NETWORK_OPTIONS:
        evaluate_detection = st.checkbox("Evaluate detection", value=True)
    else:
        evaluate_detection = network_type in MANDATORY_DETECTION_EVAL_NETWORK_OPTIONS

    load_weights = st.checkbox("Load weights", value=True)

    if load_weights:
        weights_type = st.selectbox("Weights type", WEIGHTS_TYPES, index=0)
    else:
        weights_type= None





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
    evaluate_detection=evaluate_detection,
    load_weights=load_weights,
    weights_type=weights_type,
)
storage_root = Path(storage_path)
experiment_root = expected_experiment_root(storage_path, dataset_name, current_results_dir)

left, right = st.columns([2, 1])

with left:
    st.subheader("Command Preview")
    st.code(format_command(command), language="powershell")

with right:
    st.subheader("Project")
    st.text_input("Project root", value=str(PROJECT_ROOT), disabled=True)
    st.text_input("Python", value=sys.executable, disabled=True)
    st.text_input("Expected output root", value=str(experiment_root), disabled=True)

if storage_path.strip() and not storage_root.exists():
    st.warning("The selected storage path does not exist on this machine.")

if not use_gpu:
    st.warning("Running this code may require a GPU depending on the selected network and hardware.")

cuda_required_for_selected_run = (
    run_script == "Training"
    and network_type in OPTIONAL_DETECTION_EVAL_NETWORK_OPTIONS
    and not use_gpu
)
if cuda_required_for_selected_run:
    st.error("This training mode requires CUDA in the current code.")

run_clicked = st.button(
    "Run LegoNet",
    type="primary",
    disabled=cuda_required_for_selected_run,
)

if run_clicked:
    if not storage_path.strip():
        st.error("Storage path is required.")
        st.stop()

    output_placeholder = st.empty()
    status_placeholder = st.empty()
    output_lines: list[str] = []

    status_placeholder.info("Running LegoNet...")
    process = subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    if process.stdout is not None:
        for line in process.stdout:
            output_lines.append(line)
            output_placeholder.code("".join(output_lines))

    returncode = process.wait()

    if returncode == 0:
        status_placeholder.empty()
        st.success("Run completed successfully.")
    else:
        status_placeholder.empty()
        st.error(f"Run failed with exit code {returncode}.")

st.divider()
st.subheader("Recent Result Files")

artifacts = find_recent_artifacts(experiment_root)
if not artifacts:
    st.info("No CSV or text result files found yet for the selected experiment folder.")
else:
    selected_artifact = st.selectbox(
        "Result file",
        artifacts,
        format_func=lambda path: str(path.relative_to(experiment_root)),
    )
    st.caption(str(selected_artifact))
    st.download_button(
        "Download selected file",
        data=selected_artifact.read_bytes(),
        file_name=selected_artifact.name,
    )
    st.code(read_preview(selected_artifact))
