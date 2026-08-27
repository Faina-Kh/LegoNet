# Detailed setup and GUI usage

The main README provides the shortest path to installing and running LegoNet.
This guide collects the more detailed storage and Streamlit behavior needed for
custom local, remote, or development setups.

## Storage directory

When the source checkout directory is named `Code`, LegoNet uses its parent as
the default storage directory. This keeps source files separate from downloaded
datasets, experiment results, and checkpoints:

```text
LegoNet/
├── Code/          Git repository and installed project source
├── Datasets/      Downloaded datasets and copied annotation metadata
├── ExpResults/    Training and inference results
└── checkpoints/   Downloaded pretrained weights
```

You can therefore run the quick-inference commands from `Code/` without
`--storage-path`.

For compatibility, a checkout with another directory name uses its repository
root as storage. Clone or rename the checkout as `Code`, or pass
`--storage-path` explicitly, to use the sibling-folder layout.

To keep runtime files elsewhere, supply a storage root on each command:

```bash
python scripts/run_legonet.py --storage-path /path/to/legonet-storage --help
```

or set `LEGONET_STORAGE_PATH` before running the CLI or Streamlit app.

PowerShell, current terminal:

```powershell
$env:LEGONET_STORAGE_PATH = "D:\LegoNet"
```

PowerShell, persistent for the current Windows user:

```powershell
[Environment]::SetEnvironmentVariable(
    "LEGONET_STORAGE_PATH",
    "D:\LegoNet",
    "User"
)
```

Open a new terminal after setting a persistent variable.

### Recommended workspace layout

Create the four-folder workspace by cloning the repository as `Code`:

```bash
mkdir LegoNet
cd LegoNet
git clone https://github.com/Faina-Kh/LegoNet.git Code
cd Code
```

The resulting runtime layout is:

```text
LegoNet/
├── Code/
├── Datasets/
│   ├── Embrapa WGISD/
│   │   ├── train.txt
│   │   ├── val.txt
│   │   ├── test.txt
│   │   └── classes.txt
│   └── Grapevines data/
│       ├── sub_Train/
│       │   ├── Train.csv
│       │   ├── Train_pointsOutput.csv
│       │   └── Train_Dia_Length_Color.txt
│       ├── sub_Val/
│       └── sub_Test/
├── ExpResults/
│   ├── grapes/
│   └── roots/
└── checkpoints/
    └── zenodo-21966953/
```

Dataset, result, and checkpoint paths are constructed from these directory
names. Missing directories are created automatically.

## Streamlit GUI

Install the optional GUI dependencies and start the local app with:

```bash
python -m pip install -e ".[gui]"
streamlit run apps/streamlit_app.py
```

The Storage path field starts with `LEGONET_STORAGE_PATH` when the environment
variable is set and otherwise uses the source-checkout root. It remains
editable and includes a **Browse local machine…** button. The native folder
picker opens on the machine running Streamlit and may be unavailable for remote
or headless deployments; manual path entry remains available in those
environments.

Weight loading is selected by mode, and the GUI displays only the checkpoint
paths required by that mode. Each checkpoint supports manual server-path entry
or browser-based file upload. Uploaded checkpoints are copied to a temporary
server-side directory before the LegoNet subprocess starts, so uploads work
both locally and when the browser connects to a remote Streamlit server. Manual
path entry remains useful when the checkpoint already exists on the machine
running Streamlit.

For a typical inference run:

1. Keep the source-checkout storage directory or select another location.
2. Choose `grapes` or `roots`, then choose the network and estimate type.
3. Leave **Weights loading** set to **Automatic pretrained weights
   (recommended)**. The GUI shows the selected filename, download size, cache
   status, and Zenodo record before the run starts.
4. Choose `Test` or `Val`, enable visualizations if required, inspect the
   command preview, and select **Run LegoNet**.

Leave **Download missing public dataset files automatically** enabled unless
the system must remain offline. LegoNet first checks the selected local dataset
and reuses it when complete. If files are missing, the live output shows JPEG
download progress for grapes, or ZIP download, checksum verification, and
extraction progress for roots. A manually prepared complete dataset is not
downloaded again even when this option is enabled.

The subprocess output reports when a checkpoint download begins and when its
checksum has been verified. The first run may download approximately 125–401
MiB depending on the selected model; later runs use the cached checkpoint.
Select **Full model checkpoint** or **Partial task checkpoints** to override
automatic selection with paths or uploaded `.pt`/`.pth` files. Select **Do not
load weights** to disable both local and downloaded checkpoints.

The GUI previews the exact CLI command before launching it.
