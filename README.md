# Real-Time ASL Sign Language Detector

This project detects American Sign Language alphabet signs from hand landmarks using MediaPipe and a TensorFlow/Keras MLP model.

It now includes two interfaces:

- `app.py`: OpenCV desktop webcam app for continuous real-time prediction.
- `streamlit_app.py`: Streamlit frontend for camera snapshots and image uploads.

## Project Structure

```text
AI-Sign-Language-Detector/
|-- app.py                    # OpenCV real-time desktop app
|-- streamlit_app.py          # Streamlit frontend
|-- requirements.txt
|-- data/
|   `-- dataset.csv
|-- model/
|   |-- asl_model.h5
|   |-- class_names.txt
|   `-- training_history.png
|-- src/
|   |-- data_collection.py
|   |-- model_training.py
|   |-- predict.py
|   `-- utils.py
`-- logs/
    `-- predictions.log
```

## Installation

Python 3.9 to 3.11 is recommended for the smoothest TensorFlow and MediaPipe compatibility.

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run The Streamlit Frontend

```bash
streamlit run streamlit_app.py
```

The Streamlit app supports:

- Camera snapshots through the browser.
- Image upload for `.jpg`, `.jpeg`, `.png`, and `.bmp`.
- Adjustable confidence threshold.
- Adjustable smoothing window.
- Mirrored camera mode.
- Annotated hand landmarks and top confidence chart.

## Run The OpenCV App

```bash
python app.py
```

Controls:

| Key | Action |
| --- | --- |
| `Q` | Quit |
| `R` | Reset smoothing buffer |

## Dataset And Training

Collect or import data, then train the model before running inference if the model files are missing.

```bash
python src\data_collection.py
python src\model_training.py
```

Expected generated files:

- `data/dataset.csv`
- `model/asl_model.h5`
- `model/class_names.txt`
- `model/training_history.png`

## Model Notes

- Input: 63 features from 21 hand landmarks with `x`, `y`, and `z`.
- Normalization: wrist-relative coordinates.
- Prediction smoothing: majority vote over recent confident predictions.
- Default confidence threshold: configured in `src/predict.py`.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| Model not found | Run `python src\model_training.py` or place `asl_model.h5` in `model/`. |
| Class names not found | Make sure `model/class_names.txt` exists. |
| No hand detected | Use a clear, well-lit image with one visible hand. |
| Webcam unavailable in Streamlit | Check browser camera permissions and refresh the app. |
| OpenCV webcam unavailable | Try changing the camera index in `app.py` from `0` to `1`. |
