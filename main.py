import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from google.protobuf.json_format import MessageToDict

# First step is to initialize the Hands class an store it in a variable
mp_hands = mp.solutions.hands

# Now second step is to set the hands function which will hold the landmarks points
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3)

# Last step is to set up the drawing function of hands landmarks on the image
mp_drawing = mp.solutions.drawing_utils


# Reading the sample image on which we will perform the detection
# sample_img = cv2.imread('coba.jpg')
sample_img = cv2.imread('coba2.png')

# Here we are specifing the size of the figure i.e. 10 -height; 10- width.
plt.figure(figsize = [10, 10])

# Here we will display the sample image as the output.
# plt.title("Sample Image");plt.axis('off');plt.imshow(sample_img[:,:,::-1]);plt.show()


results = hands.process(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))

if results.multi_hand_landmarks:
    
    for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):
        
        print(f'HAND NUMBER: {hand_no+1}')
        print('-----------------------')
        
        for i in range(2):

            print(f'{mp_hands.HandLandmark(i).name}:')
            print(f'{hand_landmarks.landmark[mp_hands.HandLandmark(i).value]}') 

    # Return whether it is Right or Left Hand
    i = results.multi_handedness
    label = MessageToDict(i)['classification'][0]['label']
                             
    if label == 'Left':
        print("ini telapak tangan kiri")
    if label == 'Right':
        print("ini telapak tangan kanan")

