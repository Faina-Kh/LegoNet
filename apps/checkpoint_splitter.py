"""Streamlit interface for splitting current LegoNet full checkpoints."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from legonet.checkpoint_conversion import (
    ESTIMATE_TYPES,
    PER_OBJECT_NETWORKS,
    convert_full_checkpoint,
)


def uploaded_checkpoint_path(uploaded_file) -> Path:
    """Persist an uploaded checkpoint for conversion on the Streamlit server."""
    upload_directory = Path(tempfile.gettempdir()) / "legonet_checkpoint_uploads"
    upload_directory.mkdir(parents=True, exist_ok=True)
    destination = upload_directory / Path(uploaded_file.name).name
    destination.write_bytes(uploaded_file.getbuffer())
    return destination


def choose_local_directory(initial_path: str = "") -> str:
    """Open a native folder picker on the machine running Streamlit."""
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

        options = {"title": "Choose the checkpoint output directory"}
        initial_directory = Path(initial_path).expanduser()
        if initial_directory.is_dir():
            options["initialdir"] = str(initial_directory)
        return filedialog.askdirectory(**options)
    except Exception as error:
        raise RuntimeError(
            "The local folder picker is unavailable. Enter the path manually."
        ) from error
    finally:
        if root is not None:
            root.destroy()


def browse_for_output_directory() -> None:
    """Update the output-directory field from the native folder picker."""
    try:
        selected = choose_local_directory(
            st.session_state.get("output_directory", "")
        )
    except RuntimeError as error:
        st.session_state.output_directory_dialog_error = str(error)
        return

    if selected:
        st.session_state.output_directory = selected
    st.session_state.output_directory_dialog_error = ""


def checkpoint_filename(value: str, description: str) -> str:
    """Validate a checkpoint filename entered separately from its directory."""
    name = value.strip()
    if not name:
        raise ValueError(f"{description} is required.")
    if Path(name).name != name:
        raise ValueError(f"{description} must be a filename, not a path.")
    if Path(name).suffix.lower() != ".pt":
        raise ValueError(f"{description} must end with .pt.")
    return name


st.set_page_config(page_title="LegoNet Checkpoint Splitter", page_icon="L")
st.title("LegoNet Checkpoint Splitter")
st.write(
    "Split a current per-object full-model checkpoint into detector and "
    "per-object partial weight files."
)

network_type = st.selectbox(
    "Network type",
    PER_OBJECT_NETWORKS,
)
estimate_type = st.selectbox("Estimate type", ESTIMATE_TYPES)
attribute_text = st.text_input(
    "Attribute names",
    help=(
        "Space- or comma-separated names, for example: length diameter color. "
        "Leave empty when the head module is named estimator."
    ),
)
attribute_names = [
    name for name in attribute_text.replace(",", " ").split() if name
]

manual_source = st.text_input(
    "Full weights file path",
    help="Path on the machine running Streamlit.",
)
uploaded_source = st.file_uploader(
    "Upload full weights file",
    type=["pt", "pth"],
)
source_path = (
    uploaded_checkpoint_path(uploaded_source)
    if uploaded_source is not None
    else Path(manual_source) if manual_source else None
)

st.subheader("Output")
output_directory = st.text_input(
    "Output directory",
    key="output_directory",
    help=(
        "Directory on the machine running Streamlit. The local browser is "
        "available only when Streamlit runs on this computer."
    ),
)
st.button(
    "Browse local output directory...",
    on_click=browse_for_output_directory,
)
directory_dialog_error = st.session_state.get(
    "output_directory_dialog_error",
    "",
)
if directory_dialog_error:
    st.warning(directory_dialog_error)

save_detector = st.checkbox("Save detector weights", value=True)
detector_filename = (
    st.text_input("Detector filename", value="legonet_bbox.pt")
    if save_detector
    else ""
)
per_object_filename = st.text_input(
    "Per-object filename",
    value="legonet_per_object.pt",
)
overwrite = st.checkbox("Overwrite existing output files", value=False)

if st.button("Convert checkpoint", type="primary"):
    if source_path is None:
        st.error("Select or enter a full weights file.")
    elif not output_directory.strip():
        st.error("Enter or select an output directory.")
    else:
        try:
            output_root = Path(output_directory.strip())
            detector_output = (
                output_root
                / checkpoint_filename(
                    detector_filename,
                    "Detector filename",
                )
                if save_detector
                else None
            )
            per_object_output = output_root / checkpoint_filename(
                per_object_filename,
                "Per-object filename",
            )
            detector_path, head_path = convert_full_checkpoint(
                full_weights_file=source_path,
                detector_output_file=detector_output,
                per_object_output_file=per_object_output,
                network_type=network_type,
                estimate_type=estimate_type,
                attribute_names=attribute_names,
                overwrite=overwrite,
            )
        except (FileNotFoundError, FileExistsError, TypeError, ValueError) as exc:
            st.error(str(exc))
        else:
            st.success("Checkpoint conversion completed.")
            if detector_path is not None:
                st.download_button(
                    "Download detector weights",
                    data=detector_path.read_bytes(),
                    file_name=detector_path.name,
                )
            st.download_button(
                "Download per-object weights",
                data=head_path.read_bytes(),
                file_name=head_path.name,
            )
