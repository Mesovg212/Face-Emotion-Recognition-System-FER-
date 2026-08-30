import os
import math
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


EMOTION_MAP = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised",
}


def parse_emotion_from_filename(filename: str) -> Optional[int]:
    """
    RAVDESS filename format: XX-YY-EE-II-SS-RR-AA
    We read the 3rd token (EE) as the emotion code (1-8).
    Returns class index in [0..7] or None if not parsable.
    """
    base = os.path.basename(filename)
    stem, _ = os.path.splitext(base)
    parts = stem.split("-")
    if len(parts) < 3:
        return None
    try:
        emotion_code = int(parts[2])
    except ValueError:
        return None
    if emotion_code not in EMOTION_MAP:
        return None
    return emotion_code - 1  # 0..7


def list_videos(root_dir: str, exts=(".mp4", ".avi", ".mov")) -> List[str]:
    paths = []
    for r, _, files in os.walk(root_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                paths.append(os.path.join(r, f))
    return sorted(paths)


def sample_indices(num_frames_in_video: int, num_frames_target: int) -> List[int]:
    if num_frames_in_video <= 0:
        return []
    if num_frames_in_video >= num_frames_target:
        # even sampling across the video length
        return [int(x) for x in np.linspace(0, num_frames_in_video - 1, num_frames_target)]
    # pad by repeating last frame index
    idxs = [int(x) for x in np.linspace(0, num_frames_in_video - 1, num_frames_in_video)]
    pad = [idxs[-1]] * (num_frames_target - num_frames_in_video)
    return idxs + pad


class VideoDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        num_frames: int = 32,
        image_size: int = 224,
        face_crop: bool = False,
        grayscale: bool = False,
    ):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.face_crop = face_crop
        self.grayscale = grayscale
        self.videos: List[Tuple[str, int]] = []

        all_videos = list_videos(root_dir)
        for p in all_videos:
            label = parse_emotion_from_filename(p)
            if label is not None:
                self.videos.append((p, label))

        if len(self.videos) == 0:
            raise RuntimeError(f"No labeled video files found under: {root_dir}")

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if grayscale:
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize(image_size),
                T.CenterCrop(image_size),
                T.Grayscale(num_output_channels=1),
                T.ToTensor(),
                T.Normalize(mean=[0.5], std=[0.5]),
            ])
            self.channels = 1
        else:
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize(image_size),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ])
            self.channels = 3

        # Initialize face detector if requested
        self.face_cascade = None
        if face_crop:
            try:
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
            except Exception:
                self.face_cascade = None

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx: int):
        video_path, label = self.videos[idx]
        frames = self._load_video_frames(video_path, self.num_frames)
        frames_tensor = torch.stack(frames, dim=0)  # T, C, H, W
        return frames_tensor, label

    def _load_video_frames(self, video_path: str, num_frames_target: int) -> List[torch.Tensor]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = sample_indices(total, num_frames_target)
        frames: List[torch.Tensor] = []

        for frame_idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                # fallback: append black frame if read fails
                if self.channels == 1:
                    img = np.zeros((224, 224), dtype=np.uint8)
                else:
                    img = np.zeros((224, 224, 3), dtype=np.uint8)
                frames.append(self.transform(img))
                continue

            # BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.face_cascade is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    frame = frame[y:y+h, x:x+w]

            frames.append(self.transform(frame))

        cap.release()
        return frames


def emotion_names() -> List[str]:
    return [EMOTION_MAP[i] for i in range(1, 9)]