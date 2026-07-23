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


st.set_page_config(page_title="LegoNet Checkpoint Converter", page_icon="L")
st.title("LegoNet Checkpoint Converter")
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

save_detector = st.checkbox("Save detector weights", value=True)
detector_output = (
    st.text_input("Detector output file path") if save_detector else ""
)
per_object_output = st.text_input("Per-object output file path")
overwrite = st.checkbox("Overwrite existing output files", value=False)

if st.button("Convert checkpoint", type="primary"):
    if source_path is None:
        st.error("Select or enter a full weights file.")
    elif save_detector and not detector_output:
        st.error("Enter a detector output file.")
    elif not per_object_output:
        st.error("Enter a per-object output file.")
    else:
        try:
            detector_path, head_path = convert_full_checkpoint(
                full_weights_file=source_path,
                detector_output_file=detector_output,
                per_object_output_file=per_object_output or None,
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
