import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import copy
import time
import mediapipe.python as mp
from WLASL.start_kit.preprocess import convert_frames_to_video

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.5)


def _extract_tensor(landmark_type, indices):
    arr=[]
    if not landmark_type:
        return np.zeros(len(indices) * 4)
    else:
        for ind in indices:
            p = landmark_type.landmark[ind]
            arr.append([p.x, p.y, p.z, p.visibility])
        return np.array(arr)
        # return np.array(arr).flatten()
    

def get_datapoints(results):
    #Pose Indexing: 11-16 (wrist, elbow, shoulder)
    pose_indices = [11, 12, 13, 14, 15, 16]
    left_indices = [i for i in range(21)]
    right_indices = [i for i in range(21)]

    #Face outline
    face_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 401, 361, 435, 288,
                    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 
                    132, 93, 234, 127, 162, 21, 54, 103, 67, 109] 
    #Top Lip
    face_indices += [61, 76, 185, 184, 183, 191, 40, 74, 42, 80, 39, 73, 41, 81, 37, 72, 
                    38, 82, 0, 11, 12, 13, 267, 302, 268, 312, 269, 303, 271, 311, 270, 304, 272, 310,
                    409, 408, 407, 415, 291, 308]  
    #Bottom Lip
    face_indices += [146, 77, 96, 95, 91, 90, 89, 88, 181, 180, 179, 178, 87, 86, 85, 84, 
                    14, 15, 16, 17, 317, 316, 315, 314, 402, 403, 404, 405, 318, 319, 320, 321,
                    324, 325, 307, 375, 306, 292, 62, 78]
    #Right Eyebrow
    face_indices += [336, 296, 334, 293, 300, 285, 295, 282, 283, 276]
    #Left Eyebrow
    face_indices += [70, 63, 105, 66, 107, 46, 53, 52, 65, 55]

    face = _extract_tensor(results.face_landmarks, face_indices)
    left = _extract_tensor(results.left_hand_landmarks, left_indices)
    pose = _extract_tensor(results.pose_landmarks, pose_indices)
    right = _extract_tensor(results.right_hand_landmarks, right_indices)
    
    return face, left, pose, right


def draw_datapoints_from_tensors(body_landmarks, org_image):
    image = copy.deepcopy(org_image)
    for landmark_tensor in body_landmarks:
        for keypoint in landmark_tensor:
            if keypoint.shape == ():
                continue
            x, y = (int(keypoint[0] * image.shape[1]), int(keypoint[1] * image.shape[0]))
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Draw a green circle
    return cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)


def annotate_video(video_path, save_video_path=None):
    keypoints_tensor = []
    cap = cv2.VideoCapture(video_path)
    frame_set = []
    i = 0
    while cap.isOpened():
        status, image = cap.read()

        if not status:
            continue
        
        image = cv2.flip(image, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_image)

        face, left, pose, right = get_datapoints(results)
        keypoints_tensor.append(np.concatenate([face.flatten(), left.flatten(), pose.flatten(), right.flatten()], axis=0))
        if i == 0:
            print(keypoints_tensor[0].shape)

        if save_video_path:
            frame_set.append(draw_datapoints_from_tensors([face, left, pose, right], rgb_image))
        i += 1
        if i >= 30:
            break

    if save_video_path:
        convert_frames_to_video(frame_set, save_video_path, (rgb_image.shape[1], rgb_image.shape[0]))

    cap.release()
    return np.array(keypoints_tensor)
        

def draw_datapoints(landmark_type, indices, image):
    if not landmark_type:
        print(f"Error no datapoints detected (empty landmark list)")
        return image
    
    for index in indices:
        point = landmark_type.landmark[index]
        x, y = int(point.x * image.shape[1]), int(point.y * image.shape[0])
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Draw a green circle
    return image


def _draw_landmarks(image, results):
    if results.face_landmarks:
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    return image


def mp_demo(first_frame=False, stop_frame=-1):
    cap = cv2.VideoCapture(0)
    count = 0
    while cap.isOpened():
        status, image = cap.read()

        if not status:
            continue
        
        image = cv2.flip(image, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_image)

        if (cv2.waitKey(1) & 0xFF == ord('q')) or first_frame:
            break
        if stop_frame != -1 and count < stop_frame:
            count += 1
        elif stop_frame != -1 and count >= stop_frame:
            break

        image = _draw_landmarks(image, results)
        cv2.imshow("Cam", image)

    cap.release()
    cv2.destroyAllWindows()
    return results, image


if __name__ == "__main__":
    results = mp_demo()

