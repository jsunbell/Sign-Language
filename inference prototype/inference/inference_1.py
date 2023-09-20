import mediapipe as mp
import cv2
# import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import time

def load_and_preprocess_data_inference(data, max_frames):
    # data = np.load(file_path)
    n_landmarks = 543
    n_dims = 3
    n_step = 4

    # Reshape (n_frames, n_landmarks, n_dims)
    data_reshaped = np.reshape(data, (data.shape[0], n_landmarks, n_dims))

    # Padding (frame 뒤에 추가 )
    if data_reshaped.shape[0] < max_frames:
        zero_padding = np.zeros((max_frames - data_reshaped.shape[0], n_landmarks, n_dims))
        data_reshaped = np.concatenate([data_reshaped, zero_padding], axis=0)
    else:
        data_reshaped = data_reshaped[:max_frames, :]

    # final reshape (n_step , n_frames/n_step , n_landmarks , n_dims)
    data_reshaped = np.reshape(data_reshaped, (8, 25 , n_landmarks * n_dims))
    predict_data = np.expand_dims(data_reshaped, axis=0)

    return predict_data


video_path = 'movie/열나다.avi'  # KETI_SL_0000000131.avi
output_dir = 'movie/'

mp_holistic = mp.solutions.holistic
integrated_data = []

with mp_holistic.Holistic(static_image_mode=False, model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    cap = cv2.VideoCapture(0)
    
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if ret > 190:
            print("End of video stream.")
            break
        
        current_time = time.time()

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)

        face = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark]) if results.face_landmarks else np.zeros((468, 3))
        pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 3))
        left_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))
        right_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))

        integrated_frame_data = np.concatenate([face.flatten(), pose.flatten(), left_hand.flatten(), right_hand.flatten()])
        integrated_data.append(integrated_frame_data)

        # 화면에 결과 표시
        cv2.imshow('MediaPipe Holistic Detection', cv2.resize(frame, None, fx=0.5, fy=0.5))
        if cv2.waitKey(1) == ord('q') or current_time - start_time >= 10:
            break
        
    cap.release()
    cv2.destroyAllWindows()

integrated_data = np.array(integrated_data, dtype=float)
# output_file_path = os.path.join(output_dir, 'mov.npy')
# np.save(output_file_path, integrated_data)


# load_and_preprocess_data_inference 함수를 거친 후.
model_path = "model/best_model_CNN_LSTM_v2.h5"
loaded_model = load_model(model_path)
test = load_and_preprocess_data_inference(integrated_data, 200)

final = loaded_model.predict(test)
print(np.argmax(final))