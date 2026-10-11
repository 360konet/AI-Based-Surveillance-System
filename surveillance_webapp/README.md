# Perimeter surveillance

A local web app around the CNN-LSTM model trained in the notebook. Two modes:

- **Live surveillance**: click Start surveillance, the webcam turns on, and frames are
  analyzed continuously. Incidents are flagged on screen, logged with a snapshot, and
  can be acknowledged and exported as CSV.
- **Test a clip**: upload a recorded clip and see the prediction, probabilities and the
  crops the model saw.

Pipeline: YOLOv8n person detection → person crops → ResNet-50 + LSTM → decision layer.

## How live mode works

- The browser sends about 5 frames per second (the rate the model was trained on).
- The server keeps a sliding window of the last 16 person crops and re-classifies it
  about once a second (every 5 new crops), starting once 8 crops are available.
- An incident is raised only when the same non-normal behaviour is predicted in 2
  consecutive windows, at or above the alert threshold, in a restricted zone. Repeated
  detections of the same behaviour within 10 s extend the open incident instead of
  creating a new one.
- If nobody is seen for 2 s, the window is cleared.
- These settings are at the top of `live.py`.
- On a Mac, the browser and Terminal may need camera permission:
  System Settings → Privacy & Security → Camera.

## Setup

1. Put this `surveillance_webapp` folder inside `/Users/mikie/Downloads`, next to the
   `outputs/` folder the notebook created. The app expects:

       Downloads/
         outputs/cnn_lstm.pt        <- saved by notebook Section 13
         yolov8n.pt                 <- downloaded by notebook Section 3 (re-downloads if missing)
         surveillance_webapp/
           app.py  pipeline.py  live.py  config.py  static/index.html

   If your files live elsewhere, set `MODEL_PATH` / `YOLO_PATH` (see below).

2. Install dependencies (same environment as the notebook is fine):

       pip install -r requirements.txt

3. Start the server from inside the folder:

       cd ~/Downloads/surveillance_webapp
       uvicorn app:app --port 8000

4. Open http://localhost:8000

Custom paths:

    MODEL_PATH=/path/to/cnn_lstm.pt YOLO_PATH=/path/to/yolov8n.pt uvicorn app:app --port 8000

## API

- `GET /api/status` — whether the model loaded, device, classes, zones
- `POST /api/live/start`, `POST /api/live/frame?session=…&zone=…&threshold=…` (JPEG body), `POST /api/live/stop?session=…`
- `POST /api/analyze` — multipart form: `video` (file), `zone` (`gate_area` | `perimeter_fence` | `open_area`)

Returns predicted label, per-class probabilities, alert decision, timing stats and
the person crops the model saw.

## Notes

- Preprocessing matches training exactly, including OpenCV's BGR channel order. If
  you retrain with different settings (SEQ_LEN, FRAME_RATE, classes), update the
  constants in `config.py` to match.
- Only the first 16 sampled frames containing a person are classified (same as
  training), so for long clips only roughly the first 3+ seconds with a person count.
- Zones are a dropdown stand-in for real camera metadata; edit `RESTRICTED_ZONES`
  and `ALL_ZONES` in `config.py` to change them.
