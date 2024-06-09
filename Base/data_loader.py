
import os
import cv2

import numpy as np
import tensorflow as tf
import mediapipe as mp

from typing import List
from Base.architecture import char_to_num_vocab

def load_video(path: str) -> np.ndarray:
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
    cap = cv2.VideoCapture(path)
    frames = []
    mouth_region_resized = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = np.array([(lm.x, lm.y) for lm in face_landmarks.landmark])
                min_x = int(np.min(landmarks[4:30, 0]) * frame.shape[1])
                max_x = int(np.max(landmarks[4:30, 0]) * frame.shape[1])
                min_y = int(np.min(landmarks[28:58, 1]) * frame.shape[0])
                max_y = int(np.max(landmarks[28:58, 1]) * frame.shape[0])
                x_end = frame.shape[0] // 7
                y_start = frame.shape[1] // 6
                target_shape = (140, 46)
                mouth_region = frame[min_y + y_start:max_y, min_x:max_x + x_end]
                mouth_region_resized = cv2.resize(mouth_region, target_shape)
                mouth_region_resized = cv2.cvtColor(mouth_region_resized, cv2.COLOR_RGB2GRAY)
                frames.append(mouth_region_resized)
    cap.release()
    face_mesh.close()
    frames1 = np.array(frames)
    sequence_length = 75
    if len(frames) < sequence_length:
        frames.extend([np.zeros_like(mouth_region_resized)] * (sequence_length - len(frames)))
    elif len(frames) > sequence_length:
        frames = frames[:sequence_length]
    frames_array = np.array(frames)
    mean = np.mean(frames_array, axis=(1, 2), keepdims=True)
    std = np.std(frames_array, axis=(1, 2), keepdims=True)
    normalized_frames = (frames_array - mean) / (std + 1e-6)
    return normalized_frames

def load_alignments(path: str, char_to_num) -> List[str]:
    with open(path, 'r') as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil':
            tokens = [*tokens, ' ', line[2]]
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]

def load_data(path: str):
    path = bytes.decode(path.numpy())
    file_name = path.split('/')[-1].split('.')[0]
    video_path = os.path.join('data', 't-data', 's', f'{file_name}.mpg')
    alignment_path = os.path.join('data', 't-data', 'align', f'{file_name}.align')
    frames = load_video(video_path)
    char_to_num, _ = char_to_num_vocab()
    alignments = load_alignments(alignment_path, char_to_num)
    return frames, alignments

def mappable_function(path: str) -> List[str]:
    result = tf.py_function(load_data, [path], (tf.float32, tf.int64))
    return result
