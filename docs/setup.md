# Detailed setup and GUI usage

The main README provides the shortest path to installing and running LegoNet.
This guide collects the more detailed storage and Streamlit behavior needed for
custom local, remote, or development setups.

## Verify the active environment

After installation, verify that Python imports LegoNet from the current
checkout:

```bash
python -c "import legonet; print(legonet.__file__)"
legonet --help
```

The module path should be inside the current `LegoNet/Code` directory. If the
import fails or points elsewhere, activate the intended environment and run
`python -m pip install -e .` again from `Code/`.

To see which executables the shell will use, run the following in Windows
Command Prompt:

```cmd
where python
where legonet
```

In Windows PowerShell, use:

```powershell
Get-Command python
Get-Command legonet
```

On Linux, macOS, WSL, or Git Bash, use:

```bash
command -v python
command -v legonet
```

The active environment's paths should appear first. If `legonet` is not
installed there, the shell may continue searching `PATH` and launch an older
installation from another Python or Conda environment.

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

Checkpoint selection has two steps. First choose the **Checkpoint
configuration**, which determines which model components are initialized from
weights. Depending on the selected network and whether the run performs
training or inference, the available configurations are:

- **Full model checkpoint** loads every model component from one file.
- **Separate detector and estimation-head checkpoints** loads the detector and
  per-object head from independent files.
- **Detector only; initialize estimation head from scratch** is available for
  per-object training. It loads and freezes the detector while training a new
  estimation head.
- **No checkpoint** starts without a LegoNet checkpoint where that
  configuration is supported.

Next, choose a source for every checkpoint required by the selected
configuration:

- **Automatic download (recommended)** shows the published filename, download
  size, cache status, and Zenodo record. The file is downloaded and verified
  when the run starts, or reused immediately if it is already cached.
- **User-provided file** reveals controls for entering a path on the machine
  running Streamlit or uploading a `.pt`/`.pth` file through the browser. These
  controls remain hidden for automatically selected checkpoints.

Each component has its own source choice. A separate-checkpoint configuration
can therefore mix sources—for example, a user-provided detector checkpoint and
an automatically downloaded estimation-head checkpoint. Uploaded files are
copied to a temporary server-side directory before the LegoNet subprocess
starts, so uploads work both locally and when the browser connects to a remote
Streamlit server. Manual path entry remains useful when the checkpoint already
exists on the Streamlit server.

For a typical inference run:

1. Keep the source-checkout storage directory or select another location.
2. Choose `grapes` or `roots`, then choose the network and estimate type.
3. Keep **Checkpoint configuration** set to **Full model checkpoint**, then
   keep **Full model checkpoint source** set to **Automatic download
   (recommended)**. Confirm the displayed filename, size, cache status, and
   Zenodo record.
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
MiB per selected checkpoint, depending on the model; later runs reuse cached
files. When **User-provided file** is selected, **Run LegoNet** remains disabled
until a path or uploaded file has been supplied for every required checkpoint.
Choose **No checkpoint** to disable both local and downloaded LegoNet
checkpoints where the selected training or inference configuration permits it.

The GUI previews the exact CLI command before launching it.
