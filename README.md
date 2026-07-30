# Braille text-to-speech converter

A Streamlit application that locates Braille cells with a hosted Roboflow
object-detection model, classifies each crop locally with EfficientNet-B0,
orders the predictions for reading, and back-translates the result with
Liblouis.

## Project layout

```text
backend/                 Detection, classification, and translation code
models/                  EfficientNet weights and class mapping
experiments/             Saved experiment results and evaluation artifacts
prototype_testing/       Training notebooks and alternative model pipelines
streamlit-sample-images/ Example inputs
streamlit_app.py         Streamlit application entry point
```

## Setup

The Roboflow Inference SDK currently requires Python 3.10–3.12. Create a
Python 3.12 environment and install the dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Create a private local environment file from the template:

```bash
cp .env.example .env
```

Set your real key in `.env`:

```text
ROBOFLOW_API_KEY=your_real_key
```

The `.env` file is ignored by Git. The configured model is
`braille-detection-f0rb5/10`, called through
`https://serverless.roboflow.com`. To select another deployment without
editing code, set `ROBOFLOW_MODEL_ID=project-name/version` in the process
environment.

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

If the classifier file in `models/` is a Git LFS pointer, download the real
weights:

```bash
git lfs pull
```

Run the application from the project root:

```bash
python -m streamlit run streamlit_app.py
```

The hosted model is used only to locate boxes. Roboflow class labels are
discarded; every detected crop is classified by the local EfficientNet model
before the results are sorted into reading order.
