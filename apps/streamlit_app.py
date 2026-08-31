from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from legonet.streamlit_output import (
    extract_evaluation_summary,
    separate_execution_time,
)
from legonet.pretrained import (
    ZENODO_RECORD_URL,
    checkpoint_cache_dir,
    select_published_checkpoint,
)
from legonet.datasets import default_storage_root


DATASET_OPTIONS = ("roots", "grapes")
NETWORK_OPTIONS = (
    "bbox_detection",
    "per_image_attributes",
    "per_object_counting",
    "per_object_attributes",
    "per_object_attributes_multibranch",
)
NETWORKS_OPTIONS_BY_DATASETS = {'roots': ("bbox_detection", "per_image_estimation", "per_object_attributes",
                                          "per_object_attributes_multibranch"),
                                'grapes': ("bbox_detection", "per_object_counting")
                                }
RUN_MODES = ("Inference", "Training")
VAL_SETS = ("Test", "Val")
ESTIMATE_TYPES = ("keypoints", "regression")
ESTIMATE_TYPE_VALUES = {
    "keypoints": "keypoints",
    "regression": "regression",
}
OPTIONAL_DETECTION_EVAL_NETWORK_OPTIONS = ("per_object_counting", "per_object_attributes", "per_object_attributes_multibranch")
PER_OBJECT_NETWORKS = OPTIONAL_DETECTION_EVAL_NETWORK_OPTIONS
ATTRIBUTE_CHECKPOINT_NAMES = ("length", "diameter", "color")
MANDATORY_DETECTION_EVAL_NETWORK_OPTIONS = ("bbox_detection")
ESTIMATE_SELECT_NETWORK_OPTIONS = (
    "per_image_estimation",
    "per_object_counting",
    "per_object_attributes",
)
DEFAULT_ESTIMATE_TYPE_BY_NETWORK = {
    "per_image_estimation": "keypoints",
    "per_object_attributes_multibranch": "keypoints"
}
STORAGE_PATH_STATE_KEY = "runner_storage_path"


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
            st.session_state.get(STORAGE_PATH_STATE_KEY, "")
        )
    except RuntimeError as error:
        st.session_state.storage_path_dialog_error = str(error)
        return

    if selected_path:
        st.session_state[STORAGE_PATH_STATE_KEY] = selected_path
    st.session_state.storage_path_dialog_error = ""


def persist_uploaded_weights(uploaded_file, state_key: str) -> str:
    """Persist a browser-uploaded checkpoint for the training subprocess."""
    signature = (uploaded_file.name, uploaded_file.size)
    signature_key = f"{state_key}_upload_signature"
    path_key = f"{state_key}_uploaded_path"
    existing_path = st.session_state.get(path_key)
    if (
        st.session_state.get(signature_key) == signature
        and existing_path
        and Path(existing_path).is_file()
    ):
        return existing_path

    upload_directory = Path(tempfile.gettempdir()) / "legonet_streamlit_weights"
    upload_directory.mkdir(parents=True, exist_ok=True)
    safe_name = Path(uploaded_file.name).name
    destination = upload_directory / f"{uuid.uuid4().hex}_{safe_name}"
    destination.write_bytes(uploaded_file.getbuffer())
    st.session_state[signature_key] = signature
    st.session_state[path_key] = str(destination)
    return str(destination)


def select_checkpoint_file(
    *,
    label: str,
    state_key: str,
    component: str,
    dataset_name: str,
    network_type: str,
    estimate_type: str,
    storage_path: str,
) -> tuple[bool, str]:
    """Select automatic retrieval or a user-provided checkpoint file."""
    source = st.radio(
        f"{label} source",
        ("Automatic download (recommended)", "User-provided file"),
        key=f"runner_source_{state_key}",
        horizontal=True,
    )
    automatic = source.startswith("Automatic")
    if automatic:
        try:
            checkpoint = select_published_checkpoint(
                dataset_name,
                network_type,
                estimate_type,
                component,
            )
            cached_path = checkpoint_cache_dir(storage_path) / checkpoint.filename
            cache_status = (
                "already cached"
                if cached_path.is_file()
                else "downloaded when the run starts"
            )
            st.info(
                f"{label}: {checkpoint.filename} "
                f"({checkpoint.size / (1024 * 1024):.1f} MiB); {cache_status}."
            )
            st.markdown(f"[Published checkpoint and license]({ZENODO_RECORD_URL})")
        except ValueError as error:
            st.warning(str(error))
        return True, ""

    saved_path_key = f"runner_saved_{state_key}"
    input_key = f"runner_input_{state_key}"
    if saved_path_key not in st.session_state:
        st.session_state[saved_path_key] = ""
    if input_key not in st.session_state:
        st.session_state[input_key] = st.session_state[saved_path_key]

    manual_path = st.text_input(
        f"{label} path",
        key=input_key,
        help="Enter a path that exists on the machine running Streamlit.",
    )
    if manual_path:
        st.session_state[saved_path_key] = manual_path

    uploaded_file = st.file_uploader(
        f"Browse local machine for {label.lower()}",
        type=["pt", "pth"],
        key=f"upload_{state_key}",
    )
    if uploaded_file is not None:
        selected_path = persist_uploaded_weights(uploaded_file, state_key)
        st.session_state[saved_path_key] = selected_path
        st.caption(f"Uploaded checkpoint: {uploaded_file.name}")
    else:
        selected_path = manual_path or st.session_state[saved_path_key]
        if selected_path:
            st.caption(f"Selected checkpoint: {Path(selected_path).name}")
    return False, selected_path


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
    draw_detection_overview: bool,
    draw_gt_only: bool,
    draw_individual_object_visualizations: bool,
    evaluate_detection: bool,
    checkpoint_attribute: str,
    weights_mode: str,
    download_missing_data: bool = True,
    full_weights_file: str = "",
    bbox_weights_file: str = "",
    per_object_weights_file: str = "",
) -> list[str]:
    """Build the LegoNet main.py command from GUI settings."""
    command = [
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
        "--draw-detection-overview",
        bool_arg(draw_detection_overview),
        "--draw-gt-only",
        bool_arg(draw_gt_only),
        "--draw-individual-object-visualizations",
        bool_arg(draw_individual_object_visualizations),
        "--evaluate-detection",
        bool_arg(evaluate_detection),
        "--weights-mode",
        weights_mode,
        "--download-missing-data",
        bool_arg(download_missing_data),
    ]
    if checkpoint_attribute:
        command.extend(["--checkpoint-attribute", checkpoint_attribute])
    if full_weights_file:
        command.extend(["--full-weights-file", full_weights_file])
    if bbox_weights_file:
        command.extend(["--bbox-weights-file", bbox_weights_file])
    if per_object_weights_file:
        command.extend(["--per-object-weights-file", per_object_weights_file])
    return command


def format_command(command: list[str]) -> str:
    """Format a command for display in the UI."""
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def keep_live_output_at_bottom(placeholder, update_id: int) -> None:
    """Keep only the live-output container scrolled to its newest content."""
    script = f"""
        <script>
            window.setTimeout(() => {{
                const root = window.parent.document.querySelector(
                    '.st-key-live_output_scroller'
                );
                if (!root) return;
                const elements = [root, ...root.querySelectorAll('*')];
                const scroller = elements.find((element) => {{
                    const style = window.parent.getComputedStyle(element);
                    return element.scrollHeight > element.clientHeight &&
                        (style.overflowY === 'auto' || style.overflowY === 'scroll');
                }});
                if (scroller) scroller.scrollTop = scroller.scrollHeight;
                // Force component refresh for output update {update_id}.
            }}, 0);
        </script>
    """
    with placeholder.container():
        components.html(script, height=0)


def expected_experiment_root(storage_path: str, dataset_name: str, current_results_dir: str) -> Path:
    """Return the experiment folder where main.py writes results for these settings."""
    return (
        Path(storage_path)
        / "ExpResults"
        / dataset_name
        / "Results"
        / current_results_dir
    )


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


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
MAIN_SCRIPT = PROJECT_ROOT / "scripts" / "run_legonet.py"
GPU_AVAILABLE = has_cuda_gpu()
DEFAULT_STORAGE_ROOT = default_storage_root(PROJECT_ROOT) or PROJECT_ROOT

st.set_page_config(page_title="LegoNet Runner", page_icon="L", layout="wide")

st.title("LegoNet Runner")

with st.sidebar:
    st.header("Run Settings")
    if STORAGE_PATH_STATE_KEY not in st.session_state:
        st.session_state[STORAGE_PATH_STATE_KEY] = st.session_state.get(
            "storage_path",
            os.environ.get(
                "LEGONET_STORAGE_PATH",
                str(DEFAULT_STORAGE_ROOT),
            ),
        )

    storage_path = st.text_input(
        "Storage path",
        key=STORAGE_PATH_STATE_KEY,
        help=(
            "Select the LegoNet project folder or another storage folder. "
            "LegoNet stores Datasets, ExpResults, and downloaded checkpoints "
            "inside the selected folder. For the recommended layout, select "
            "the outer LegoNet folder that contains Code."
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

    st.session_state.setdefault("runner_dataset_name", DATASET_OPTIONS[0])
    dataset_name = st.selectbox(
        "Dataset",
        DATASET_OPTIONS,
        key="runner_dataset_name",
    )

    available_network_types = NETWORKS_OPTIONS_BY_DATASETS[dataset_name]
    if st.session_state.get("runner_network_type") not in available_network_types:
        st.session_state.runner_network_type = available_network_types[0]
    selected_network_type = st.selectbox(
        "Network type",
        available_network_types,
        key="runner_network_type",
    )

    run_script = st.selectbox(
        "Run mode",
        RUN_MODES,
        key="runner_run_mode",
    )

    if run_script == 'Inference':
        val_set = st.selectbox(
            "Validation set",
            VAL_SETS,
            key="runner_validation_set",
        )
    else:
        val_set = "Val"

    use_gpu = st.checkbox(
        "Use GPU",
        value=GPU_AVAILABLE,
        key="runner_use_gpu",
    )
    if use_gpu:
        gpu_num = st.text_input(
            "GPU number",
            value="0",
            key="runner_gpu_number",
        )
    else:
        gpu_num = ""

    st.header("Advanced")
    if selected_network_type in ESTIMATE_SELECT_NETWORK_OPTIONS:
        selected_estimate_type = st.selectbox(
            "Estimate type",
            ESTIMATE_TYPES,
            key="runner_estimate_type",
            help=(
                "Choose how attributes are estimated: from detected keypoints "
                "or directly with a regression model."
            ),
        )
        estimate_type = ESTIMATE_TYPE_VALUES[selected_estimate_type]
    else:
        estimate_type = DEFAULT_ESTIMATE_TYPE_BY_NETWORK.get(
            selected_network_type,
            "regression",
        )

    network_type = selected_network_type

    if run_script == "Training":
        num_of_epochs = st.number_input(
            "Number of epochs",
            min_value=1,
            value=300,
            step=1,
            key="runner_num_epochs",
        )
    else:
        num_of_epochs = 0

    results_dir_context = (selected_network_type, run_script, val_set)
    estimate_suffix = (
        ""
        if selected_network_type == "bbox_detection"
        else "_KP" if estimate_type == "keypoints" else "_Reg"
    )
    if st.session_state.get("runner_results_dir_context") != results_dir_context:
        results_suffix = "Training" if run_script == "Training" else val_set
        st.session_state.runner_results_dir = (
            selected_network_type + estimate_suffix + "_" + results_suffix
        )
        st.session_state.runner_results_dir_context = results_dir_context
    current_results_dir = st.text_input(
        "Current results dir",
        key="runner_results_dir",
        help=(
            "You can edit this run's results folder name. The folder will be "
            "created at: Storage path/ExpResults/Dataset/Current results dir."
        ),
    )

    have_gt = st.checkbox(
        "Have ground truth",
        value=True,
        key="runner_have_gt",
    )
    download_missing_data = st.checkbox(
        "Download missing public dataset files automatically",
        value=True,
        key="runner_download_missing_data",
        help=(
            "For grapes, downloads only the JPEG images referenced by the "
            "tracked annotations. For roots, downloads and verifies the "
            "published Zenodo archive."
        ),
    )
    to_draw = st.checkbox(
        "Draw visualizations",
        value=False,
        key="runner_draw_visualizations",
    )
    if to_draw and network_type in PER_OBJECT_NETWORKS:
        individual_context = dataset_name
        if (
            st.session_state.get("runner_individual_visualizations_context")
            != individual_context
        ):
            st.session_state.runner_draw_individual_object_visualizations = (
                dataset_name == "roots"
            )
            st.session_state.runner_individual_visualizations_context = (
                individual_context
            )
        draw_individual_object_visualizations = st.checkbox(
            "Draw separate box visualization for each object",
            key="runner_draw_individual_object_visualizations",
            help=(
                "Save separate full-image GT-box and predicted-box views. "
                "Predicted crops and keypoint heatmaps are always retained. "
                "The dataset default is enabled for roots and disabled for "
                "grapes."
            ),
        )
        draw_detection_overview = st.checkbox(
            "Draw detection overview",
            value=True,
            key="runner_draw_detection_overview",
            help="Save full-image GT/prediction overlays; disable for dense images.",
        )
        if have_gt:
            draw_gt_only = st.checkbox(
                "Draw GT-only images",
                value=False,
                key="runner_draw_gt_only",
                help="Save annotation-debugging images without predictions.",
            )
        else:
            draw_gt_only = False
    else:
        draw_individual_object_visualizations = dataset_name == "roots"
        draw_detection_overview = False
        draw_gt_only = False
    if not have_gt:
        evaluate_detection = False
    elif network_type in OPTIONAL_DETECTION_EVAL_NETWORK_OPTIONS:
        evaluate_detection = st.checkbox(
            "Evaluate detection",
            value=True,
            key="runner_evaluate_detection",
        )
    else:
        evaluate_detection = network_type in MANDATORY_DETECTION_EVAL_NETWORK_OPTIONS

    checkpoint_attribute = ""
    if (
        run_script == "Training"
        and network_type
        in ("per_object_attributes", "per_object_attributes_multibranch")
    ):
        checkpoint_attribute = st.selectbox(
            "Best-epoch attribute",
            ATTRIBUTE_CHECKPOINT_NAMES,
            key="runner_checkpoint_attribute",
            format_func=str.title,
            help=(
                "Select the attribute whose validation error is minimized. "
                "Counting always minimizes count relative error."
            ),
        )

    full_weights_file = ""
    bbox_weights_file = ""
    per_object_weights_file = ""
    missing_user_checkpoint = False
    if network_type in PER_OBJECT_NETWORKS:
        if run_script == "Training":
            weight_layouts = (
                "Detector only; initialize estimation head from scratch",
                "Full model checkpoint",
                "Separate detector and estimation-head checkpoints",
            )
        else:
            weight_layouts = (
                "Full model checkpoint",
                "Separate detector and estimation-head checkpoints",
                "No checkpoint",
            )
    else:
        weight_layouts = ("Full model checkpoint", "No checkpoint")

    layout_context = (network_type, run_script)
    if st.session_state.get("runner_weight_layout_context") != layout_context:
        st.session_state.runner_weight_layout = weight_layouts[0]
        st.session_state.runner_weight_layout_context = layout_context
    weight_layout = st.selectbox(
        "Checkpoint configuration",
        weight_layouts,
        key="runner_weight_layout",
        help="Choose which model components to initialize from checkpoints.",
    )

    if weight_layout == "No checkpoint":
        weights_mode = "none"
    elif weight_layout == "Full model checkpoint":
        automatic, full_weights_file = select_checkpoint_file(
            label="Full model checkpoint",
            state_key="full_weights_file",
            component="full",
            dataset_name=dataset_name,
            network_type=network_type,
            estimate_type=estimate_type,
            storage_path=storage_path,
        )
        # In explicit ``full`` mode, an omitted path asks the resolver to
        # download the published full checkpoint; a supplied path is used as-is.
        weights_mode = "full"
        missing_user_checkpoint = not automatic and not full_weights_file
    elif weight_layout.startswith("Detector only"):
        st.caption(
            "The detector is frozen and the per-object estimation head is "
            "initialized from scratch."
        )
        automatic, bbox_weights_file = select_checkpoint_file(
            label="Bounding-box detector checkpoint",
            state_key="bbox_weights_file",
            component="bbox",
            dataset_name=dataset_name,
            network_type=network_type,
            estimate_type=estimate_type,
            storage_path=storage_path,
        )
        weights_mode = "detector_only"
        missing_user_checkpoint = not automatic and not bbox_weights_file
    else:
        automatic_bbox, bbox_weights_file = select_checkpoint_file(
            label="Bounding-box detector checkpoint",
            state_key="bbox_weights_file",
            component="bbox",
            dataset_name=dataset_name,
            network_type=network_type,
            estimate_type=estimate_type,
            storage_path=storage_path,
        )
        automatic_head, per_object_weights_file = select_checkpoint_file(
            label="Per-object estimation-head checkpoint",
            state_key="per_object_weights_file",
            component="head",
            dataset_name=dataset_name,
            network_type=network_type,
            estimate_type=estimate_type,
            storage_path=storage_path,
        )
        weights_mode = "partial"
        missing_user_checkpoint = (
            (not automatic_bbox and not bbox_weights_file)
            or (not automatic_head and not per_object_weights_file)
        )





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
    draw_detection_overview=draw_detection_overview,
    draw_gt_only=draw_gt_only,
    draw_individual_object_visualizations=(
        draw_individual_object_visualizations
    ),
    evaluate_detection=evaluate_detection,
    checkpoint_attribute=checkpoint_attribute,
    weights_mode=weights_mode,
    download_missing_data=download_missing_data,
    full_weights_file=full_weights_file,
    bbox_weights_file=bbox_weights_file,
    per_object_weights_file=per_object_weights_file,
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
if missing_user_checkpoint:
    st.error("Select a checkpoint file for every user-provided checkpoint source.")

run_clicked = st.button(
    "Run LegoNet",
    type="primary",
    disabled=cuda_required_for_selected_run or missing_user_checkpoint,
)

if "run_output" not in st.session_state:
    st.session_state.run_output = ""
if "run_status" not in st.session_state:
    st.session_state.run_status = None
if "evaluation_summary" not in st.session_state:
    st.session_state.evaluation_summary = ""
if "execution_time" not in st.session_state:
    st.session_state.execution_time = ""

# Migrate output retained by a Streamlit session from an earlier app version.
st.session_state.run_output, retained_execution_time = separate_execution_time(
    st.session_state.run_output
)
if retained_execution_time:
    st.session_state.execution_time = retained_execution_time

status_placeholder = st.empty()
live_output_container = st.container(
    height=500,
    key="live_output_scroller",
)
output_placeholder = live_output_container.empty()
scroll_trigger = st.empty()

if st.session_state.run_output:
    output_placeholder.code(st.session_state.run_output)
if st.session_state.run_status == "success":
    status_placeholder.success("Run completed successfully.")
elif st.session_state.run_status == "failed":
    status_placeholder.error("The previous run failed.")

if run_clicked:
    if not storage_path.strip():
        st.error("Storage path is required.")
        st.stop()

    output_lines: list[str] = []
    output_segment: list[str] = []
    summary_lines: list[str] = []
    output_update_count = 0
    progress_placeholder = None
    progress_label = None
    st.session_state.run_output = ""
    st.session_state.evaluation_summary = ""
    st.session_state.execution_time = ""
    st.session_state.run_status = "running"
    output_placeholder.empty()

    status_placeholder.info("Running LegoNet...")
    process = subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={
            **os.environ,
            "LEGONET_PROGRESS_PROTOCOL": "1",
            "LEGONET_STREAMLIT_SUMMARIES": "1",
        },
    )

    if process.stdout is not None:
        for line in process.stdout:
            if line.startswith("__LEGONET_PROGRESS__\t"):
                _, label, current_text, total_text = line.rstrip("\r\n").split(
                    "\t", maxsplit=3
                )
                current = int(current_text)
                total = int(total_text)
                if progress_placeholder is None or label != progress_label:
                    progress_placeholder = live_output_container.empty()
                    progress_label = label
                    output_placeholder = None
                    output_segment.clear()
                progress_placeholder.progress(
                    min(current / total, 1.0) if total else 1.0,
                    text=f"{label} {current}/{total}",
                )
                if current == 1 or current % 5 == 0 or current >= total:
                    keep_live_output_at_bottom(
                        scroll_trigger,
                        output_update_count + current,
                    )
                continue
            line, execution_time = separate_execution_time(line)
            if execution_time:
                st.session_state.execution_time = execution_time
                continue
            output_lines.append(line)
            if output_placeholder is None:
                output_placeholder = live_output_container.empty()
            output_segment.append(line)
            output_update_count += 1
            st.session_state.run_output = "".join(output_lines)
            line_summary = extract_evaluation_summary(line)
            if line_summary:
                summary_lines.append(line_summary)
                st.session_state.evaluation_summary = "\n".join(summary_lines)
            if output_update_count == 1 or output_update_count % 5 == 0:
                output_placeholder.code("".join(output_segment))
                keep_live_output_at_bottom(
                    scroll_trigger,
                    output_update_count,
                )

    returncode = process.wait()
    if output_placeholder is not None:
        output_placeholder.code("".join(output_segment))
    keep_live_output_at_bottom(scroll_trigger, output_update_count + 1)

    if returncode == 0:
        st.session_state.run_status = "success"
        status_placeholder.success("Run completed successfully.")
    else:
        st.session_state.run_status = "failed"
        status_placeholder.error(f"Run failed with exit code {returncode}.")

if st.session_state.evaluation_summary:
    st.subheader("Evaluation Summary")
    st.code(
        st.session_state.evaluation_summary,
        language="text",
        wrap_lines=True,
    )
if st.session_state.execution_time:
    st.code(st.session_state.execution_time, language="text")

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
