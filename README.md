# ASL Sign Language Detector

A Python application for recognizing static American Sign Language alphabet signs using MediaPipe hand landmarks and a TensorFlow/Keras neural network.

The project includes a Streamlit interface for photo-based prediction and a realtime OpenCV webcam mode for continuous recognition.

## Features

- Detects 24 static ASL letters: `A-Z` except `J` and `Z`
- Photo detection from camera snapshots or uploaded images
- Realtime webcam detection through an OpenCV window
- MediaPipe hand landmark extraction
- Wrist-relative landmark normalization
- Lightweight Keras MLP classifier
- Confidence thresholding and majority-vote smoothing
- Model report page with accuracy, F1 scores, confusion matrix, training curves, and dataset statistics

## Why J and Z Are Excluded

The letters `J` and `Z` are motion-based ASL signs. This project currently classifies single-frame static hand poses, so those two letters are not included in the model classes.

## Tech Stack

- Python
- Streamlit
- OpenCV
- MediaPipe
- TensorFlow/Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## Project Structure

```text
Sign Language Recognizer/
|-- app.py                       # OpenCV realtime desktop app
|-- streamlit_app.py             # Streamlit app entry point
|-- shared.py                    # Shared Streamlit/runtime helpers
|-- model_report.py              # Model evaluation report generator
|-- import_dataset.py            # Dataset import helper
|-- manage_dataset.py            # Dataset management helper
|-- requirements.txt
|-- pages/
|   |-- 1_Home.py
|   |-- 2_Photo_Detection.py
|   |-- 3_Realtime_Webcam.py
|   `-- 4_Model_Report.py
|-- src/
|   |-- data_collection.py
|   |-- model_training.py
|   |-- predict.py
|   `-- utils.py
|-- data/
|   |-- dataset.csv              # Local/generated dataset, ignored by Git
|   `-- webcam.csv               # Local webcam samples, ignored by Git
|-- model/
|   |-- asl_model.h5             # Trained model, ignored by Git
|   |-- class_names.txt
|   `-- training_history.png
`-- images/
    `-- Hand_Signs.png
```

## Setup

Python 3.9 to 3.11 is recommended for the smoothest TensorFlow and MediaPipe compatibility.

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run The Streamlit App

```bash
streamlit run streamlit_app.py
```

The Streamlit app includes:

- Home page
- Photo detection page
- Realtime webcam launch page
- Model performance report page

## Run The OpenCV Webcam App

```bash
python app.py
```

Controls:

| Key | Action |
| --- | --- |
| `Q` | Quit webcam window |
| `R` | Reset smoothing buffer |

## Dataset

The full dataset files are intentionally ignored by Git:

- `data/dataset.csv`
- `data/webcam.csv`
- `data/*.bak`

This keeps the GitHub repository lightweight and prevents large generated data from being committed repeatedly.

To run the project on another device, download or copy the dataset files separately and place them in:

```text
data/dataset.csv
data/webcam.csv
```


## Dataset Format

Each row contains one labelled hand-sign sample:

```text
label,x0,y0,z0,x1,y1,z1,...,x20,y20,z20
```

- `label`: ASL letter
- `x/y/z`: MediaPipe coordinates for 21 hand landmarks
- Total input features per sample: `21 * 3 = 63`
- Coordinates are normalized relative to the wrist landmark

## Train The Model

Train using the main dataset:

```bash
python src\model_training.py
```

Expected generated files:

```text
model/asl_model.h5
model/class_names.txt
model/training_history.png
```

## Generate Model Report

```bash
python model_report.py
```

The report evaluates the trained model on a held-out 20% test split and shows:

- Test accuracy
- Test loss
- Per-letter precision, recall, and F1 score
- Confusion matrix
- Dataset sample counts


```
## Troubleshooting

| Problem | Fix |
| --- | --- |
| Missing dependency | Run `pip install -r requirements.txt` |
| Model not found | Run `python src\model_training.py` or copy `model/asl_model.h5` into the project |
| Dataset not found | Place `dataset.csv` inside the `data/` folder |
| Class names not found | Ensure `model/class_names.txt` exists or retrain the model |
| No hand detected | Use a clear, well-lit image with one visible hand |
| Webcam unavailable | Check camera permissions or try another camera index |
| TensorFlow/MediaPipe install issues | Use Python 3.9 to 3.11 |

