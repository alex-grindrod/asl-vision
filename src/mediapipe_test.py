import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe.python as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.5)


def draw_landmarks(image, results):
    if results.face_landmarks:
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    return image

def get_datapoints(results):
    #Pose Indexing: 11-16 (wrist, elbow, shoulder)
    pose = np.array([[p.x, p.y, p.z, p.visibility] for p in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    left = np.array([[p.x, p.y, p.z] for p in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    right = np.array([[p.x, p.y, p.z] for p in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    """
    Face Indexing:
        -  left eyebrow: 70, 63, 105, 66, 107      46, 53, 52, 65, 55
        - right eyebrow: 336, 296, 334, 293, 300   285, 295, 282, 283, 276
    """
    face = np.array([[p.x, p.y, p.z] for p in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    return pose, left, right, face

def draw_datapoints(landmark_type, indices, image):
    print(len(indices))
    if not landmark_type:
        print(f"Error no datapoints detected (empty landmark list)")
        return image
    
    for index in indices:
        point = landmark_type.landmark[index]
        x, y = int(point.x * image.shape[1]), int(point.y * image.shape[0])
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Draw a green circle
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

        image = draw_landmarks(image, results)
        cv2.imshow("Cam", image)

    cap.release()
    cv2.destroyAllWindows()
    return results, image




if __name__ == "__main__":
    results = mp_demo()
    # get_datapoints(results)

