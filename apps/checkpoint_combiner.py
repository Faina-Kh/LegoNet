"""Streamlit interface for combining LegoNet partial checkpoints."""

from __future__ import annotations

import io
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from legonet.checkpoint_conversion import (
    ESTIMATE_TYPES,
    PER_OBJECT_NETWORKS,
    combine_partial_checkpoints,
)


class Tee:
    """Write console text to both the terminal and an in-memory GUI log."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text: str) -> int:
        """Write text to every configured stream."""
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        """Flush every configured stream."""
        for stream in self.streams:
            stream.flush()


def uploaded_checkpoint_path(uploaded_file, role: str) -> Path:
    """Persist an uploaded checkpoint on the Streamlit server."""
    upload_directory = Path(tempfile.gettempdir()) / "legonet_checkpoint_uploads"
    upload_directory.mkdir(parents=True, exist_ok=True)
    destination = upload_directory / f"{role}_{Path(uploaded_file.name).name}"
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
        options = {"title": "Choose the full-checkpoint output directory"}
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
            st.session_state.get("combine_output_directory", "")
        )
    except RuntimeError as error:
        st.session_state.combine_directory_dialog_error = str(error)
        return
    if selected:
        st.session_state.combine_output_directory = selected
    st.session_state.combine_directory_dialog_error = ""


def selected_input(label: str, role: str) -> Path | None:
    """Return an uploaded checkpoint or a server-local path."""
    manual = st.text_input(f"{label} path")
    uploaded = st.file_uploader(
        f"Upload {label.lower()}",
        type=["pt", "pth"],
        key=f"upload_{role}",
    )
    if uploaded is not None:
        return uploaded_checkpoint_path(uploaded, role)
    return Path(manual) if manual.strip() else None


def checkpoint_filename(value: str) -> str:
    """Validate the full-checkpoint output filename."""
    name = value.strip()
    if not name:
        raise ValueError("Full-model filename is required.")
    if Path(name).name != name:
        raise ValueError("Full-model filename must be a filename, not a path.")
    if Path(name).suffix.lower() != ".pt":
        raise ValueError("Full-model filename must end with .pt.")
    return name


st.set_page_config(page_title="LegoNet Checkpoint Combiner", page_icon="L")
st.title("LegoNet Checkpoint Combiner")
st.write(
    "Combine compatible detector and per-object partial weights into one "
    "validated full-model checkpoint."
)

network_type = st.selectbox("Network type", PER_OBJECT_NETWORKS)
estimate_type = st.selectbox("Estimate type", ESTIMATE_TYPES)
default_attribute_names = (
    "length diameter color"
    if network_type in (
        "per_object_attributes",
        "per_object_attributes_multibranch",
    )
    else ""
)
attribute_text = st.text_input(
    "Attribute names",
    value=default_attribute_names,
    help=(
        "Space- or comma-separated names, for example: length diameter color. "
        "Leave empty when the head module is named estimator."
    ),
)
attribute_names = [
    name for name in attribute_text.replace(",", " ").split() if name
]

detector_source = selected_input("Detector weights file", "detector")
head_source = selected_input("Per-object weights file", "per_object")

st.subheader("Output")
output_directory = st.text_input(
    "Output directory",
    key="combine_output_directory",
)
st.button(
    "Browse local output directory...",
    on_click=browse_for_output_directory,
)
dialog_error = st.session_state.get("combine_directory_dialog_error", "")
if dialog_error:
    st.warning(dialog_error)
full_filename = st.text_input(
    "Full-model filename",
    value="legonet_full_model.pt",
)
overwrite = st.checkbox("Overwrite existing output file", value=False)

if st.button("Combine checkpoints", type="primary"):
    if detector_source is None:
        st.error("Select or enter the detector weights file.")
    elif head_source is None:
        st.error("Select or enter the per-object weights file.")
    elif not output_directory.strip():
        st.error("Enter or select an output directory.")
    else:
        console_output = io.StringIO()
        try:
            with redirect_stdout(
                Tee(sys.stdout, console_output)
            ), redirect_stderr(Tee(sys.stderr, console_output)):
                output = Path(output_directory.strip()) / checkpoint_filename(
                    full_filename
                )
                saved = combine_partial_checkpoints(
                    detector_weights_file=detector_source,
                    per_object_weights_file=head_source,
                    full_output_file=output,
                    network_type=network_type,
                    estimate_type=estimate_type,
                    attribute_names=attribute_names,
                    overwrite=overwrite,
                )
        except (
            FileNotFoundError,
            FileExistsError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            if console_output.getvalue():
                st.subheader("Conversion log")
                st.code(console_output.getvalue(), language="text")
            st.error(str(exc))
        else:
            if console_output.getvalue():
                st.subheader("Conversion log")
                st.code(console_output.getvalue(), language="text")
            st.success("Full-model checkpoint created.")
            st.download_button(
                "Download full-model weights",
                data=saved.read_bytes(),
                file_name=saved.name,
            )
