import sys
sys.path.append(".")

import cv2
import json
import time
from moviepy.editor import VideoFileClip
from WLASL.start_kit import preprocess
from tqdm import tqdm
from collections import deque
from pathlib import Path
from src.utilities import ASLConfig


CONFIG = ASLConfig()

JSON_FILE_PATH = Path("WLASL/start_kit/WLASL_v0.3.json")
RAW_VIDEOS_PATH = CONFIG.raw_dir

FRAME_CAP = CONFIG.frame_cap
SUBSET = CONFIG.subset
RESOLUTION = CONFIG.resolution
INTERPOLATION = CONFIG.interpolation
PROCESSED_DIR = CONFIG.processed_dir
VARIANT_SPLIT = CONFIG.variant_split


def create_directories(gloss, instance):
    path = PROCESSED_DIR / gloss
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path


def get_video_path(instance):
    """
        Returns Video Path that an instance refers to if it exists 
        otherwise returns None
    """
    if "youtube" in instance["url"]:
        _, yt_id = instance["url"].split("=")
        yt_id += ".mp4"
        for video in RAW_VIDEOS_PATH.glob(yt_id):
            return RAW_VIDEOS_PATH / yt_id
        
    for video in RAW_VIDEOS_PATH.glob(instance["video_id"] + ".mp4"):
        return RAW_VIDEOS_PATH / (instance["video_id"] + ".mp4")

    return None

# def get_video_path(instance):
#     if "youtube" in instance["url"]:
#         _, yt_id = instance["url"].split("=")
#         yt_id += ".mp4"
#         for video in RAW_VIDEOS_PATH.glob(yt_id):
#             return RAW_VIDEOS_PATH / yt_id
        
#     for video in RAW_VIDEOS_PATH.glob(instance["video_id"] + ".mp4"):
#         return RAW_VIDEOS_PATH / (instance["video_id"] + ".mp4")

#     return None


def can_open_video(video_path):
    """
    Checks to see if video file is playable/openable
    (If it wasn't already obvious by the name of the function)
    """
    video_path = str(video_path)
    try:
        clip = VideoFileClip(video_path)
        return True
    except:
        return False


def expand_frames(frames, frame_cap=30):
    """
    Expands Frame Count to frame_cap by duplicating frames from middle --> out
    """
    first_half = deque(frames[:len(frames) // 2])
    second_half = deque(frames[len(frames) // 2 :])
    frame_set = deque()
    first = True
    while len(first_half) + len(second_half) + len(frame_set) != frame_cap:
        if not first_half and not second_half:
            return list(frame_set)
        if first:
            if not first_half:
                return list(frame_set) + [second_half[0], second_half[0]]
            frame = first_half.pop()
            frame_set.extendleft([frame, frame])
        else:
            if not second_half:
                return [first_half[0], first_half[0]] + list(frame_set) 
            frame = second_half.popleft()
            frame_set.extend([frame, frame])
        first = not first
    return list(first_half) + list(frame_set) + list(second_half)


def _select_frames(frames, frames_to_keep):
    """
    Evenly select frames to keep
    """
    frames_to_keep = int(frames_to_keep)
    step = len(frames) / frames_to_keep
    return [frames[int(round(i * step))] for i in range(frames_to_keep)]


def downscale_frames(frames, frame_cap=30):
    """
    Reduce framerate following semi-normal distribution
    Select greater frames towards to middle to keep
    """
    interval = len(frames) // 3
    beg_frames = frames[:interval]
    mid_frames = frames[interval : interval*2]
    end_frames = frames[interval*2:]

    beg_frames = _select_frames(beg_frames, frame_cap*0.3)
    mid_frames = _select_frames(mid_frames, frame_cap*0.4)
    end_frames = _select_frames(end_frames, frame_cap*0.3)

    return beg_frames + mid_frames + end_frames


def set_frames(frames, frame_cap=30):
    """
    Force list of frames into specified frame_count 
    (Handles Upscale and Downscale)
    """
    if len(frames) < frame_cap:
        while len(frames) < frame_cap:
            frames = expand_frames(frames, frame_cap=frame_cap)
            if len(frames) > frame_cap:
                raise ValueError(f"TOO MANY FRAMSSSS: {len(frames)}")
    elif len(frames) > frame_cap:
        frames = downscale_frames(frames, frame_cap=frame_cap)
    return frames


def resize_frames(frames, resolution):
    """
    Resize list of frames to specified resolution
    """
    return [cv2.resize(frame, resolution, interpolation=INTERPOLATION) for frame in frames]


# def crop_frames(frames, bbox):
#     """
#     Crops set of frames to focus on signer/person in video
#     """
#     x,y,w,h = bbox
#     height, width, _ = frames[0].shape

#     padding_frame = np.zeros((height, width, 3), dtype=np.uint8)
#     cropped_frames = []

#     for frame in frames:
#         cropped_frame = frame[y:h, x:w]
#         frame = np.copy(padding_frame)
        
#         #Calculate Centering Coordinates
#         pad_x = max(0, (width - abs(x - w)) // 2)
#         pad_y = max(0, (height - abs(y - h)) // 2)

#         #Apply Crop
#         frame[pad_y:pad_y + abs(y - h), pad_x:pad_x + abs(x - w)] = cropped_frame

#         cropped_frames.append(frame)
#     return cropped_frames


if __name__ == "__main__":

    start = time.time()
    with open(JSON_FILE_PATH, 'r') as metadata_file:
        annotations = json.load(metadata_file) 
        
    for i in tqdm(range(SUBSET), desc="Processing"):
        word = annotations[i]
        gloss = word["gloss"]
        instances = word["instances"]

        path = PROCESSED_DIR / gloss
        if path.exists():
            print(f"Word: {gloss} exists --> skipping")
            continue

        for instance in instances:
            video_path = get_video_path(instance)
            if not can_open_video(video_path):
                continue

            save_dir = create_directories(gloss, instance)
            frames = preprocess.video_to_frames(video_path)
            if len(frames) == 0:
                print(video_path)
                print(instance["video_id"])
                exit()
            frames = set_frames(frames, frame_cap=FRAME_CAP)
            frames = resize_frames(frames, resolution=RESOLUTION)

            # Commented cuz some videos have the wrong bbox dimensions
            # frames = crop_frames(frames, instance["bbox"]) 

            save_dir = save_dir / (instance["video_id"] + ".mp4")

            preprocess.convert_frames_to_video(frames, save_dir, RESOLUTION)
    print(time.time() - start)