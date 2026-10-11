"""
Inference pipeline for the web app.

Everything here mirrors the training notebook (Sections 3, 4, 6, 11) so that a clip
uploaded through the browser is processed EXACTLY the way the training clips were:

  YOLOv8n person detection -> largest-person crop per sampled frame -> 224x224
  -> /255 -> ImageNet normalise -> ResNet-50 features -> LSTM -> 4-class softmax
  -> rule-based decision layer (behaviour + zone -> alert / no alert)

Important: the notebook fed OpenCV frames straight into the model, which means the
crops are in BGR channel order, not RGB. We keep BGR here on purpose. "Fixing" it to
RGB would make inference differ from training and quietly hurt accuracy.
"""

import os
import time
import threading

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

from config import (ALL_ZONES, CLASSES, FRAME_RATE, IMAGENET_MEAN, IMAGENET_STD,  # noqa: F401
                    IMG_SIZE, PERSON_CLASS_ID, RESTRICTED_ZONES, SEQ_LEN, decide_alert)


# ---- Model definition (identical to notebook Section 6) ---------------------------
class CNNLSTM(nn.Module):
    def __init__(self, num_classes=4, hidden_size=256, lstm_layers=1):
        super().__init__()
        # weights=None: the trained ResNet weights come from cnn_lstm.pt, so there's
        # no need to download ImageNet weights every time the server starts.
        resnet = models.resnet50(weights=None)
        resnet.fc = nn.Identity()
        self.cnn = resnet
        self.lstm = nn.LSTM(input_size=2048, hidden_size=hidden_size,
                            num_layers=lstm_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        feats = self.cnn(x).view(b, t, -1)
        out, _ = self.lstm(feats)
        return self.classifier(out[:, -1, :])


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class SurveillancePipeline:
    def __init__(self, model_path, yolo_path):
        from ultralytics import YOLO  # imported here so startup errors are clearer

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Trained model not found at '{model_path}'. Set MODEL_PATH to the "
                "cnn_lstm.pt file the notebook saved in Section 13."
            )

        self.device = pick_device()
        self.detector = YOLO(yolo_path)  # downloads yolov8n.pt automatically if missing

        self.model = CNNLSTM(num_classes=len(CLASSES))
        state = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

        self.norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        self.lock = threading.Lock()  # one inference at a time (MPS isn't happy otherwise)
        self.model_path = model_path

        # Feature of an all-zero crop: what a padding frame looks like after the CNN.
        self.pad_feature = self.embed_crop(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))

    # -- Live mode helpers -----------------------------------------------------------
    # The ResNet is frozen and in eval mode, so each frame's features don't depend on
    # the other frames. Embedding each crop once as it arrives, then running only the
    # LSTM over the last 16, gives the same output as model(clip) at a fraction of the cost.
    def embed_crop(self, crop_bgr):
        x = torch.from_numpy(crop_bgr.astype(np.float32) / 255.0).permute(2, 0, 1)
        x = self.norm(x).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.model.cnn(x)[0]

    def classify_features(self, feats):
        feats = list(feats)[:SEQ_LEN]
        feats += [self.pad_feature] * (SEQ_LEN - len(feats))  # trailing pad, as in training
        with torch.no_grad():
            out, _ = self.model.lstm(torch.stack(feats).unsqueeze(0))
            logits = self.model.classifier(out[:, -1, :])
            return torch.softmax(logits, dim=1)[0].cpu().numpy()

    # -- Section 3 ----------------------------------------------------------------
    def detect_persons(self, frame):
        results = self.detector.predict(frame, classes=[PERSON_CLASS_ID], verbose=False,
                                        device=self.device.type)
        boxes = []
        for r in results:
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                boxes.append((int(x1), int(y1), int(x2), int(y2), float(b.conf[0])))
        return boxes

    # -- Section 4 (plus bookkeeping the UI shows) -----------------------------------
    def extract_person_sequence(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("OpenCV couldn't open this video. Try re-exporting it as .mp4 (H.264).")
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(int(round(native_fps / FRAME_RATE)), 1)

        crops, previews, det_times = [], [], []
        sampled = 0
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                sampled += 1
                t0 = time.perf_counter()
                boxes = self.detect_persons(frame)
                det_times.append(time.perf_counter() - t0)
                if boxes:
                    x1, y1, x2, y2, conf = max(boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
                    crop = frame[max(y1, 0):y2, max(x1, 0):x2]
                    if crop.size > 0:
                        crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
                        crops.append(crop)
                        if len(previews) < SEQ_LEN:
                            # annotated full frame for the first few kept frames
                            vis = frame.copy()
                            cv2.rectangle(vis, (x1, y1), (x2, y2), (40, 170, 230), 3)
                            previews.append({"crop": crop, "frame": vis,
                                             "conf": conf, "t": idx / native_fps})
            idx += 1
        cap.release()

        stats = {
            "native_fps": round(float(native_fps), 2),
            "duration_s": round(total_frames / native_fps, 2) if total_frames else None,
            "frames_sampled": sampled,
            "frames_with_person": len(crops),
            "avg_detection_ms": round(1000 * float(np.mean(det_times)), 2) if det_times else None,
        }

        if not crops:
            return None, previews, stats

        frames = np.array(crops, dtype=np.float32) / 255.0
        if len(frames) >= SEQ_LEN:
            frames = frames[:SEQ_LEN]
            stats["padded_frames"] = 0
        else:
            pad_n = SEQ_LEN - len(frames)
            pad = np.zeros((pad_n, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
            frames = np.concatenate([frames, pad], axis=0)
            stats["padded_frames"] = pad_n
        return frames, previews, stats

    # -- Sections 6/8 inference + Section 11 decision --------------------------------
    def analyze(self, video_path, zone_id):
        with self.lock:
            t_start = time.perf_counter()
            seq, previews, stats = self.extract_person_sequence(video_path)

            if seq is None:
                return {"status": "no_person", "stats": stats, "previews": previews,
                        "zone": zone_id}

            clip = torch.from_numpy(seq).permute(0, 3, 1, 2).float()      # (T,3,H,W)
            clip = torch.stack([self.norm(f) for f in clip]).unsqueeze(0)  # (1,T,3,H,W)

            t0 = time.perf_counter()
            with torch.no_grad():
                probs = torch.softmax(self.model(clip.to(self.device)), dim=1)[0].cpu().numpy()
            stats["classifier_ms"] = round(1000 * (time.perf_counter() - t0), 2)
            stats["total_ms"] = round(1000 * (time.perf_counter() - t_start), 2)

            label = CLASSES[int(probs.argmax())]
            return {
                "status": "ok",
                "label": label,
                "confidence": float(probs.max()),
                "probabilities": {c: float(p) for c, p in zip(CLASSES, probs)},
                "zone": zone_id,
                "zone_restricted": zone_id in RESTRICTED_ZONES,
                "alert": decide_alert(label, zone_id),
                "stats": stats,
                "previews": previews,
            }
