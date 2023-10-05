import mediapipe as mp
import cv2
# import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import csv
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

#################################################################################
# CSV 파일 경로 설정
csv_file_path = 'metadata/meta_word_v4.csv'
target_row = 419

# CSV 파일 열기
with open(csv_file_path, 'r', encoding='utf-8') as file:
    # CSV 파일을 읽기 위한 CSV 리더 생성
    csv_reader = csv.reader(file)
    
    # CSV 내용을 저장할 빈 리스트 생성
    csv_list = []
    
    # rdr를 순회하면서 419행까지 읽기
    for row_number, row in enumerate(csv_reader):
        if row_number > 0:
            csv_list.append([int(row[12]), row[8]])
            if row_number == target_row:
                # 19행일 때 break.
                break

csv_dic = dict(csv_list)
print(csv_dic)
# print(len(csv_dic))

##################################################################################

# 녹화된 영상일 경우,
# video_path = 'movie/기절하다.avi'
video_path = 'movie_test/test4.mkv'
output_dir = 'movie/'

mp_holistic = mp.solutions.holistic
integrated_data = []

#########################################################################################################
# # 그림 그리기를 위한 drawing_utils 모듈 초기화
# mp_drawing = mp.solutions.drawing_utils
# connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
#########################################################################################################

with mp_holistic.Holistic(static_image_mode=False, model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    cap = cv2.VideoCapture(video_path)
    
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        # print("..........")
        if not ret:
            print("End of video stream.")
            break
        
#########################################################################################################
        #     # Holistic 수행
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # results = holistic.process(frame_rgb)

        # if results.face_landmarks:
        #     # 얼굴 랜드마크 그리기
        #     mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)              

        # if results.left_hand_landmarks:
        #     # 왼손 랜드마크 그리기
        #     mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # if results.right_hand_landmarks:
        #     # 오른손 랜드마크 그리기
        #     mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # if results.pose_landmarks:
        #     # 포즈(몸) 랜드마크 그리기
        #     mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
#########################################################################################################

        current_time = time.time()
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)

        face = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark]) if results.face_landmarks else np.zeros((468, 3))
        pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 3))
        left_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))
        right_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))

        integrated_frame_data = np.concatenate([face.flatten(), pose.flatten(), left_hand.flatten(), right_hand.flatten()])
        integrated_data.append(integrated_frame_data)
        # print(integrated_data)

        # 화면에 결과 표시
        cv2.imshow('MediaPipe Holistic Detection', cv2.resize(frame, None, fx=0.5, fy=0.5))
        # if cv2.waitKey(1) == ord('q'):
        #     break
        if cv2.waitKey(1) == ord('q') or current_time - start_time >= 20:
            break
        
    cap.release()
    cv2.destroyAllWindows()

integrated_data = np.array(integrated_data, dtype=float)
# output_file_path = os.path.join(output_dir, 'mov.npy')
# np.save(output_file_path, integrated_data)


# load_and_preprocess_data_inference 함수를 거친 후.
model_path = "model/CNN_LSTM_419_ver1.h5"
loaded_model = load_model(model_path)
test = load_and_preprocess_data_inference(integrated_data, 200)

final = loaded_model.predict(test)
i = np.argmax(final)
# print(np.argmax(final))
print(f'label : {i}', f'phrase : {csv_dic[i]}')