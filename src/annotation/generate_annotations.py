import sys
sys.path.append('.')

import json
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
from src.annotation.mp_annotate import annotate_video
from pathlib import Path
from src.utilities import ASLConfig

CONFIG = ASLConfig()

PROCESSED_DIR = CONFIG.processed_dir
FRAME_CAP = CONFIG.frame_cap
JSON_FILE = PROCESSED_DIR / "_word_ids.json"

MODEL_TYPE = CONFIG.model_type
SUBSET = CONFIG.subset


def generate_priority_list():
    """
    Generates priority list by favoring words that have more ASL video instances
    """

    vid_frequencies = []
    for dir in PROCESSED_DIR.iterdir():
        if dir.is_dir():
            vid_frequencies.append([sum(1 for video in dir.iterdir() if video.is_file()), dir.name])
    vid_frequencies.sort(reverse=True)

    data = {word[1] : ind for ind, word in enumerate(vid_frequencies)}

    with open(str(JSON_FILE), 'w') as file:
        json.dump(data, file, indent=4)


def generate_word_annotations(word, model_type=None, save_dir=None):
    """
    Annotates each video in a word folder
    Saves the annotations in a .npy file
    Annotation : Tensor of shape -> (30 frames, 192 mediapipe keypoints)
    """
    word_path = PROCESSED_DIR / word
    for vid in word_path.iterdir():
        vid = vid.name
        vid_id, ext = vid.split('.')
        if ext == "mp4":
            npy_path = word_path / f"{vid_id}.npy"

            if npy_path.exists() and not save_dir:
                continue
            
            save_video_path = save_dir / word / f"{vid_id}.mp4" if save_dir else None

            vid_tensor = annotate_video(str(word_path / vid), frame_cap=FRAME_CAP, model_type=model_type, save_video_path=save_video_path)
            # print(vid_tensor.shape)
            np.save(str(npy_path), vid_tensor)


def clear_annotations():
    """
    Delete all .npy annotation files
    .npy files hold a tensor which contains the mediapipe annotations for the videos
    """
    for word_path in PROCESSED_DIR.rglob("*.npy"):   
        try:
            word_path.unlink()
        except Exception as e:
            print(f"Failed to Delete {word_path} because of: {e}")
    exit()         


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-c', '--clear', action='store_true', help='Enable wandb (default: False)')
    args = parser.parse_args()

    if args.clear:
        clear_annotations(PROCESSED_DIR)
    
    if not PROCESSED_DIR.exists():
        PROCESSED_DIR.mkdir(parents=True)
    
    if not JSON_FILE.exists():
        generate_priority_list()
    
    with open(str(JSON_FILE), 'r') as file:
        word_priority_dict = json.load(file)
    
    for ind, word in tqdm(enumerate(word_priority_dict), desc="Annotating"):
        generate_word_annotations(word, model_type=MODEL_TYPE)
        if ind >= SUBSET:
            break
