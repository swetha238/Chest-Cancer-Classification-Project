# Chest CT Scan Classification (TensorFlow · DVC · MLflow · Flask)

End‑to‑end ML pipeline to classify chest CT scans using transfer learning (VGG16), with reproducible stages, parameterization, and a simple web app for inference.

## Key Features
- Modular pipeline: Data Ingestion → Base Model Prep → Training → Evaluation
- Reproducible runs via DVC (`dvc.yaml`) and centralized params (`params.yaml`)
- MLflow‑ready evaluation step for metrics logging
- Flask app (`app.py`) for /train and /predict endpoints
- Clean `src/` package (`cnnClassifier`) with editable install

## Tech Stack
- TensorFlow/Keras, NumPy, Pandas, Matplotlib, Seaborn
- DVC for pipeline orchestration and artifacts
- MLflow (optional) for experiment tracking
- Flask + CORS for serving predictions


# Chest CT Scan Classification (TensorFlow · DVC · MLflow · Flask)

End‑to‑end ML pipeline to classify chest CT scans using transfer learning (VGG16), with reproducible stages, parameterization, and a simple web app for inference.

## Key Features
- Modular pipeline: Data Ingestion → Base Model Prep → Training → Evaluation
- Reproducible runs via DVC (`dvc.yaml`) and centralized params (`params.yaml`)
- MLflow‑ready evaluation step for metrics logging
- Flask app (`app.py`) for /train and /predict endpoints
- Clean `src/` package (`cnnClassifier`) with editable install

## Tech Stack
- TensorFlow/Keras, NumPy, Pandas, Matplotlib, Seaborn
- DVC for pipeline orchestration and artifacts
- MLflow (optional) for experiment tracking
- Flask + CORS for serving predictions

## Project Structure
```
DS-project
├─ app.py                     # Flask app
├─ main.py                    # Orchestrates all stages
├─ dvc.yaml                   # DVC stages
├─ params.yaml                # Hyperparameters
├─ requirements.txt           # Python dependencies
├─ templates/
│  └─ index.html              # Web UI
├─ config/
│  └─ config.yaml             # Paths, data sources, artifacts
└─ src/
   └─ cnnClassifier/
      ├─ components/          # data_ingestion, prepare_base_model, trainer, eval
      ├─ config/              # configuration manager
      ├─ constants/           # CONFIG_FILE_PATH, PARAMS_FILE_PATH
      ├─ entity/              # dataclasses for configs
      ├─ pipeline/            # stage_01..04 + prediction pipeline
      └─ utils/               # yaml/json helpers, logging utils
```

## Configuration
- `params.yaml` controls hyperparameters (image size, epochs, batch size, augmentation, LR, weights).
- `config/config.yaml` defines artifact paths and dataset source (Google Drive URL). Example:
```yaml
artifacts_root: artifacts
data_ingestion:
  root_dir: artifacts/data_ingestion
  source_URL: "https://drive.google.com/file/d/<YOUR_FILE_ID>/view?usp=sharing"
  local_data_file: artifacts/data_ingestion/data.zip
  unzip_dir: artifacts/data_ingestion
prepare_base_model:
  root_dir: artifacts/prepare_base_model
  base_model_path: artifacts/prepare_base_model/base_model.h5
  updated_base_model_path: artifacts/prepare_base_model/updated_model.h5
training:
  root_dir: artifacts/training
  trained_model_path: artifacts/training/model.h5
```

## Setup
```powershell
# Create and activate environment
conda create -n ds_env python=3.10 -y
conda activate ds_env

# Install project and dependencies
python -m pip install --upgrade pip
pip install -e .
pip install -r requirements.txt
```

## Train (End‑to‑End)
```powershell
# Runs all stages: ingestion → base model → train → evaluate
python main.py

# Or with DVC
dvc init          # first time (or: dvc init --no-scm)
dvc repro
```

## Serve (Flask)
```powershell
python app.py     # http://127.0.0.1:8080
```
- GET `/` → web UI (`templates/index.html`)
- POST `/train` → triggers `python main.py`
- POST `/predict` → body `{ "image": "<base64 string>" }`, returns predicted class

Note: Ensure `src/cnnClassifier/pipeline/prediction.py` loads `artifacts/training/model.h5` (produced by training).

## Pipeline Details
- Data Ingestion: downloads zip from Google Drive (gdown), extracts to `artifacts/data_ingestion/Chest-CT-Scan-data` (folder‑per‑class format).
- Prepare Base Model: loads VGG16 (imagenet weights), adds classification head, compiles with SGD + categorical crossentropy.
- Training: Keras generators with optional augmentation; saves model to `artifacts/training/model.h5`.
- Evaluation: computes loss/accuracy on validation split; writes `scores.json`. MLflow logging prepared (URI set; call can be enabled).

## Troubleshooting
- ModuleNotFoundError (`cnnClassifier`): run `pip install -e .` in the active env.
- DVC error “not inside repo”: run `dvc init` (or `dvc init --no-scm`) in project root.
- Prediction cannot find model: train first (`python main.py`) or point to the correct model path.

## What This Demonstrates
- Production‑style, parameterized, and reproducible ML pipeline
- Separation of concerns across components, config, entities, and pipelines
- Serving path from training artifacts to an API endpoint for real‑time inference
- Hooks for experiment tracking (MLflow) and data/artifact management (DVC)
