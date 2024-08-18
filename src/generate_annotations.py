import json
from pathlib import Path

DATASET_PATH = Path("data/processed_unsplit_30")
JSON_FILE = DATASET_PATH / "_word_ids.json"

def generate_priority_list():
    vid_frequencies = []
    for dir in DATASET_PATH.iterdir():
        if dir.is_dir():
            vid_frequencies.append([sum(1 for video in dir.iterdir() if video.is_file()), dir.name])
    vid_frequencies.sort(reverse=True)

    data = {word[1] : ind for ind, word in enumerate(vid_frequencies)}

    print("making file")
    with open(str(JSON_FILE), 'w') as file:
        json.dump(data, file, indent=4)

if __name__ == "__main__":
    generate_priority_list()