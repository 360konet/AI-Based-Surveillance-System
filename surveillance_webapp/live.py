"""
Live surveillance: turns a stream of webcam frames into behaviour assessments and incidents.

How it relates to training
--------------------------
Training clips were sampled at 5 fps, and the first 16 frames containing a person were
classified. Live, the browser sends ~5 frames per second, and we keep a sliding window
of the most recent 16 person crops. Every STRIDE new crops (about once a second) the
window is classified again.

Turning predictions into incidents
----------------------------------
A single window can be wrong, so an incident is only raised when:
  * the decision layer says "alert" (non-normal behaviour in a restricted zone),
  * the confidence is at or above the operator's threshold, and
  * the same behaviour is predicted CONFIRM_WINDOWS times in a row.
While the same behaviour keeps being detected, the open incident is extended rather than
a new one being logged, so one event produces one log entry.
"""

import base64
import itertools
import threading
import time
from collections import deque

import cv2

from config import CLASSES, IMG_SIZE, SEQ_LEN, decide_alert

MIN_FRAMES = 8          # classify once at least half the window holds real crops
STRIDE = 5              # re-classify every 5 new crops (~1 s at 5 fps)
CONFIRM_WINDOWS = 2     # consecutive alerting windows needed to raise an incident
MERGE_GAP_S = 10        # same behaviour within this gap extends the open incident
RESET_AFTER_S = 2.0     # clear the window when nobody has been seen for this long

_incident_ids = itertools.count(1)


def _jpeg_b64(img, width=480):
    if img.shape[1] > width:
        img = cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode() if ok else None


class LiveSession:
    def __init__(self, pipeline):
        self.p = pipeline
        self.feats = deque(maxlen=SEQ_LEN)
        self.since_classify = 0
        self.last_person_t = None
        self.streak_label, self.streak = None, 0
        self.latest = None
        self.open_incidents = {}   # label -> incident dict
        self.frames = 0
        self.started = time.time()
        self.last_seen = time.time()
        self.lock = threading.Lock()

    def _reset_window(self):
        self.feats.clear()
        self.since_classify = 0
        self.streak_label, self.streak = None, 0
        self.latest = None

    def process(self, frame, zone, threshold):
        with self.lock, self.p.lock:
            now = time.time()
            self.last_seen = now
            self.frames += 1
            h, w = frame.shape[:2]

            t0 = time.perf_counter()
            boxes = self.p.detect_persons(frame)
            det_ms = 1000 * (time.perf_counter() - t0)

            main = None
            if boxes:
                main = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                x1, y1, x2, y2, _ = main
                crop = frame[max(y1, 0):y2, max(x1, 0):x2]
                if crop.size > 0:
                    self.feats.append(self.p.embed_crop(cv2.resize(crop, (IMG_SIZE, IMG_SIZE))))
                    self.since_classify += 1
                    self.last_person_t = now
            elif self.last_person_t is None or now - self.last_person_t > RESET_AFTER_S:
                self._reset_window()

            cls_ms, new_incident, updated_incident = None, None, None
            if len(self.feats) >= MIN_FRAMES and self.since_classify >= STRIDE:
                self.since_classify = 0
                t0 = time.perf_counter()
                probs = self.p.classify_features(self.feats)
                cls_ms = 1000 * (time.perf_counter() - t0)

                i = int(probs.argmax())
                label, conf = CLASSES[i], float(probs[i])
                alerting = decide_alert(label, zone) and conf >= threshold

                if alerting and label == self.streak_label:
                    self.streak += 1
                elif alerting:
                    self.streak_label, self.streak = label, 1
                else:
                    self.streak_label, self.streak = None, 0

                self.latest = {
                    "label": label,
                    "confidence": conf,
                    "probabilities": {c: float(p) for c, p in zip(CLASSES, probs)},
                    "alerting": alerting,
                    "confirmed": alerting and self.streak >= CONFIRM_WINDOWS,
                    "streak": self.streak,
                }

                if self.latest["confirmed"]:
                    inc = self.open_incidents.get(label)
                    if inc and now - inc["last_t"] <= MERGE_GAP_S and inc["zone"] == zone:
                        inc["last_t"] = now
                        inc["peak_confidence"] = max(inc["peak_confidence"], conf)
                        updated_incident = {k: inc[k] for k in ("id", "last_t", "peak_confidence")}
                    else:
                        snap = frame.copy()
                        if main:
                            cv2.rectangle(snap, main[:2], main[2:4], (24, 35, 180), 4)
                        inc = {
                            "id": next(_incident_ids),
                            "label": label,
                            "zone": zone,
                            "first_t": now,
                            "last_t": now,
                            "confidence": conf,
                            "peak_confidence": conf,
                            "snapshot": _jpeg_b64(snap),
                        }
                        self.open_incidents[label] = inc
                        new_incident = inc

            return {
                "frame_size": [w, h],
                "persons": [list(b) for b in boxes],
                "main": list(main) if main else None,
                "window": len(self.feats),
                "window_needed": MIN_FRAMES,
                "assessment": self.latest,
                "new_incident": new_incident,
                "updated_incident": updated_incident,
                "timing": {"detection_ms": round(det_ms, 1),
                           "classifier_ms": round(cls_ms, 1) if cls_ms else None},
                "server_time": now,
            }
