"""Shared constants (copied from the training notebook) and the decision layer."""

CLASSES = ["intrusion", "loitering", "anomalous", "normal"]  # order matters (label_map)
FRAME_RATE = 5      # sampled fps
SEQ_LEN = 16        # frames per sequence fed to the LSTM
IMG_SIZE = 224      # ResNet-50 input size
PERSON_CLASS_ID = 0
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

RESTRICTED_ZONES = {"gate_area", "perimeter_fence"}
ALL_ZONES = ["gate_area", "perimeter_fence", "open_area"]


def decide_alert(behavior_label, zone_id):
    """Decision layer from notebook Section 11: normal never alerts; the other three
    classes alert only in restricted zones."""
    if behavior_label == "normal":
        return False
    return zone_id in RESTRICTED_ZONES
