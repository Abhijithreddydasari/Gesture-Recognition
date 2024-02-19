from pickle import FALSE
from typing import Sequence
import cv2 as cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import pyautogui
import mediapipe as mp
import screen_brightness_control as sbc
from pycaw.pycaw import AudioUtilities, ISimpleAudioVolume
import webbrowser
import time

#mapping action to realtime control
def hbbr(index,mode):
    if(index==0):
        sbc.set_brightness(sbc.get_brightness()[0]-20) 
    if(index==1):
        sbc.set_brightness(sbc.get_brightness()[0]+20) 
    if(index==2 or index==3):
        if(mode<3):
           return 
        sessions = AudioUtilities.GetAllSessions()
        for session in sessions:
            volume = session._ctl.QueryInterface(ISimpleAudioVolume)
            if session.Process and session.Process.name() == "Spotify.exe":
                val=volume.GetMasterVolume()
                if(index==2):
                    if(val<=0.8):
                        volume.SetMasterVolume(val+0.2, None)
                if(index==3):
                    if(val>=0.2):
                        volume.SetMasterVolume(val-0.2, None)
                    
    if(index==4):
        if(mode<3):
            return 
        webbrowser.open('https://youtu.be/-AXetJvTfU0?t=43')
    if(index==5):
        if(mode<3):
            return
        os.system("shutdown /l")
        return 

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

def draw_styled_landmarks(image, results):
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                              )
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                              )

def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])


# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data')

# Actions that we try to detect
actions = np.array(['swipe left','swipe right','thumbs up','thumbs down','ok','sleep'])
#actions = np.array(['up','down','love'])

# Thirty videos worth of data
no_sequences = [120,120,180,180,180,240]

# Videos are going to be 20 frames in length
sequence_length = 20

# Folder start
start_folder = 30

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

label_map = {label:num for num, label in enumerate(actions)}

print(label_map)

sequences, labels = [], []
for i,action in enumerate(actions):
    for sequence in range(no_sequences[i]):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense,SimpleRNN

model = Sequential()
model.add(SimpleRNN(64, return_sequences=True, activation='tanh', input_shape=(20,126)))
model.add(SimpleRNN(128, return_sequences=True, activation='tanh', dropout=0.5))
model.add(SimpleRNN(69, return_sequences=False, activation='tanh', dropout=0.5))
model.add(Dense(64, activation='tanh'))
model.add(Dense(32, activation='tanh'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

#model.fit(X_train, y_train, epochs=1500, verbose=2)
#model.save('d:\phn313-project\pythonapplication1')

import tensorflow
model=tensorflow.keras.models.load_model('D:\PHN313-Project\PythonApplication1') # Location of model

from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
yhat = model.predict(X_test)
ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()
print(multilabel_confusion_matrix(ytrue, yhat))
print(accuracy_score(ytrue, yhat))

colors = [(245,117,16), (117,245,16), (16,117,245), (245,167,16), (117,145,16), (206,117,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
       # cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

    return output_frame


# 1. New detection variables
sequence = []
sentence = []
threshold = 0.8

cap = cv2.VideoCapture(0)
# Set mediapipe model

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    flag=False
    while cap.isOpened():
   
        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        print(results)

        # Draw landmarks
        draw_styled_landmarks(image, results)

        # 2. Prediction logic
        
        keypoints = extract_keypoints(results) 
  
        if(keypoints[0]!=0 or keypoints[10]!=0 or keypoints[-10]!=0 or keypoints[-1]!=0):
            sequence.append(keypoints)

        if len(sequence) == 23:
            mode=np.zeros(6) 
            for i in range(3):
                tempseq= sequence[i:i+20]
                res = model.predict(np.expand_dims(tempseq, axis=0))[0]
                print(actions[np.argmax(res)])
                rr=np.argmax(res)
                mode[rr]+=1
            index=0
            max_mode=0
            for i in range(len(mode)):
                if(mode[i]>max_mode):
                    max_mode=mode[i]
                    index=i
            hbbr(index,max_mode)
            sequence=[]
            
        #3. Viz logic
        if res[np.argmax(res)] > threshold:
            if len(sentence) > 0:
                if actions[np.argmax(res)] != sentence[-1]:
                    sentence.append(actions[np.argmax(res)])
            else:
                sentence.append(actions[np.argmax(res)])

        if len(sentence) > 5:
            sentence = sentence[-5:]

        # Viz probabilities
        #image = prob_viz(res, actions, image, colors)

        #cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        #cv2.putText(image, ' '.join(sentence), (3,30),
        #            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()