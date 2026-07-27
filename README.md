# Braille text-to-speech converter

A Streamlit application that detects Braille cells with YOLO11, classifies
them with EfficientNet-B0, orders them for reading, and back-translates the
result with Liblouis.

## Project layout

```text
backend/                 Runtime recognition and translation code
models/                  YOLO, EfficientNet, and class-mapping files
experiments/             Saved experiment results and evaluation artifacts
prototype_testing/       Training notebooks and alternative model pipelines
streamlit-sample-images/ Example inputs
streamlit_app.py         Streamlit application entry point
```

## Setup

Install the Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Liblouis is a system dependency. Install it for your platform:

```bash
# macOS
brew install liblouis

# Debian/Ubuntu
sudo apt install liblouis-bin
```

On Windows, install Liblouis and set either `LIBLOUIS_HOME` to its installation
directory or `LIBLOUIS_TRANSLATE` to the full path of `lou_translate.exe`.
Those environment variables can also override discovery on macOS and Linux.

If the files in `models/` are Git LFS pointers, download the real weights:

```bash
git lfs pull
```

Run the application from the project root:

```bash
streamlit run streamlit_app.py
```
