import cv2 as cv2
import numpy as np
import os
import mediapipe as mp
import time 


folder = "sleep"
for count, filename in enumerate(os.listdir(folder)):
    dst = f"{str(count+120)}"
    src =f"{folder}/{filename}"  # foldername/filename, if .py file is outside folder
    dst =f"{folder}/{dst}"
         
    # rename() function will
    # rename all the files
    os.rename(src, dst)

#mp_holistic = mp.solutions.holistic # Holistic model
#mp_drawing = mp.solutions.drawing_utils # Drawing utilities

#def mediapipe_detection(image, model):
#    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
#    image.flags.writeable = False                  # Image is no longer writeable
#    results = model.process(image)                 # Make prediction
#    image.flags.writeable = True                   # Image is now writeable
#    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
#    return image, results

#def draw_landmarks(image, results):
#    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
#    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

#def draw_styled_landmarks(image, results):
#    # Draw left hand connections
#    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
#                              mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
#                              mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
#                              )
#    # Draw right hand connections
#    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
#                              mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
#                              mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
#                              )

#def extract_keypoints(results):
#    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
#    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
#    return np.concatenate([lh, rh])


## Path for exported data, numpy arrays
#DATA_PATH = os.path.join('Training Data1')

## Actions that we try to detect
#actions = np.array(['sleep'])

## Thirty videos worth of data
#no_sequences = [120,60,120,120,60]

## Videos are going to be 20 frames in length
#sequence_length = 20

## Folder start
#start_folder = 30

#for i,action in enumerate(actions):
#    for sequence in range(no_sequences[i]):
#        try:
#            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
#        except:
#            pass


#cap = cv2.VideoCapture(0)
## Set mediapipe model
#with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

#    # NEW LOOP
#    # Loop through actions
#    for i,action in enumerate(actions):
#        # Loop through sequences aka videos
#        for sequence in range(no_sequences[i]):
#            # Loop through video length aka sequence length
#            for frame_num in range(sequence_length):

#                # Read feed
#                ret, frame = cap.read()

#                # Make detections
#                image, results = mediapipe_detection(frame, holistic)
#                #                 print(results)

#                # Draw landmarks
#                draw_styled_landmarks(image, results)

#                # NEW Apply wait logic
#                if frame_num == 0:
#                    cv2.putText(image, 'STARTING COLLECTION', (120,200),
#                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
#                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12),
#                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
#                    # Show to screen
#                    cv2.imshow('OpenCV Feed', image)
#                    cv2.waitKey(1000)
#                else:
#                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12),
#                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
#                    # Show to screen
#                    cv2.imshow('OpenCV Feed', image)

#                # NEW Export keypoints
#                keypoints = extract_keypoints(results)
#                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
#                np.save(npy_path, keypoints)

#                # Break gracefully
#                if cv2.waitKey(10) & 0xFF == ord('q'):
#                    break

#        time.sleep(5)
#    cap.release()
#    cv2.destroyAllWindows()

#cap.release()
#cv2.destroyAllWindows()
