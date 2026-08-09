"""Streamlit interface for removing one module from a checkpoint."""

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

from legonet.checkpoint_conversion import remove_checkpoint_module


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


def uploaded_checkpoint_path(uploaded_file) -> Path:
    """Persist an uploaded checkpoint on the Streamlit server."""
    upload_directory = Path(tempfile.gettempdir()) / "legonet_checkpoint_uploads"
    upload_directory.mkdir(parents=True, exist_ok=True)
    destination = upload_directory / f"module_removal_{Path(uploaded_file.name).name}"
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
        options = {"title": "Choose the cleaned-checkpoint output directory"}
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
            st.session_state.get("removal_output_directory", "")
        )
    except RuntimeError as error:
        st.session_state.removal_directory_dialog_error = str(error)
        return
    if selected:
        st.session_state.removal_output_directory = selected
    st.session_state.removal_directory_dialog_error = ""


def checkpoint_filename(value: str) -> str:
    """Validate the cleaned-checkpoint output filename."""
    name = value.strip()
    if not name:
        raise ValueError("Output filename is required.")
    if Path(name).name != name:
        raise ValueError("Output filename must be a filename, not a path.")
    if Path(name).suffix.lower() not in (".pt", ".pth"):
        raise ValueError("Output filename must end with .pt or .pth.")
    return name


st.set_page_config(page_title="LegoNet Checkpoint Module Remover", page_icon="L")
st.title("LegoNet Checkpoint Module Remover")
st.write(
    "Remove one named module and all of its parameters from a checkpoint. "
    "The source checkpoint is never modified."
)

manual_source = st.text_input(
    "Weights file path",
    help="Path on the machine running Streamlit.",
)
uploaded_source = st.file_uploader(
    "Upload weights file",
    type=["pt", "pth"],
)
source_path = (
    uploaded_checkpoint_path(uploaded_source)
    if uploaded_source is not None
    else Path(manual_source) if manual_source.strip() else None
)
module_name = st.text_input(
    "Module name to remove",
    value="find_2",
    help="Use the module prefix shown in the checkpoint, for example find_2.",
)

st.subheader("Output")
output_directory = st.text_input(
    "Output directory",
    key="removal_output_directory",
)
st.button(
    "Browse local output directory...",
    on_click=browse_for_output_directory,
)
dialog_error = st.session_state.get("removal_directory_dialog_error", "")
if dialog_error:
    st.warning(dialog_error)
output_filename = st.text_input(
    "Output filename",
    value="legonet_without_find_2.pt",
)
overwrite = st.checkbox("Overwrite existing output file", value=False)

if st.button("Remove module", type="primary"):
    if source_path is None:
        st.error("Select or enter a weights file.")
    elif not module_name.strip():
        st.error("Enter the module name to remove.")
    elif not output_directory.strip():
        st.error("Enter or select an output directory.")
    else:
        console_output = io.StringIO()
        try:
            with redirect_stdout(
                Tee(sys.stdout, console_output)
            ), redirect_stderr(Tee(sys.stderr, console_output)):
                output = Path(output_directory.strip()) / checkpoint_filename(
                    output_filename
                )
                saved = remove_checkpoint_module(
                    weights_file=source_path,
                    output_file=output,
                    module_name=module_name,
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
                st.subheader("Removal log")
                st.code(console_output.getvalue(), language="text")
            st.error(str(exc))
        else:
            if console_output.getvalue():
                st.subheader("Removal log")
                st.code(console_output.getvalue(), language="text")
            st.success(f"Module {module_name.strip()!r} was removed.")
            st.download_button(
                "Download cleaned weights",
                data=saved.read_bytes(),
                file_name=saved.name,
            )
