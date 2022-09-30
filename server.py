import os
import shutil
import time
from fastapi import FastAPI, File, UploadFile,HTTPException  ,Request
# from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt


# First step is to initialize the Hands class an store it in a variable
mp_hands = mp.solutions.hands

# Now second step is to set the hands function which will hold the landmarks points
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3)

# Last step is to set up the drawing function of hands landmarks on the image
mp_drawing = mp.solutions.drawing_utils

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/")
async def image(image: UploadFile = File(...)):
    content_type = image.content_type
    if(content_type != "image/jpeg" and content_type != "image/png") :
      raise HTTPException(status_code=404,  detail="Fail bukan foto")
    image_name = image.filename
    with open("temp/"+image_name, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
    time.sleep(2)
    sample_img = cv2.imread("temp/"+image_name)
    sample_img = cv2.flip(sample_img, 1)
    results = hands.process(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))
    if not results.multi_hand_landmarks:
      raise HTTPException(status_code=404, detail="Foto harus ada telapak tangan")
        
    if len(results.multi_handedness) > 1:
      raise HTTPException(status_code=404, detail="Hanya satu telapak tangan yang bisa diramal")

    tangan =  results.multi_handedness[0].classification[0].label
    if(tangan == 'Right'):
      raise HTTPException(status_code=404, detail="Hanya Tangan Kiri Yang Bisa Diramal")

    thumb = None
    pinky = None
    for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):
      thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
      pinky = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x
    cek_tapak_tangan = thumb > pinky
    if(cek_tapak_tangan == False):
      raise HTTPException(status_code=404, detail="Sila foto telapak tangan kiri anda")

    if os.path.exists("temp/"+image_name):
      os.remove("temp/"+image_name)

    return {"message": "lakukan ramalan"}  


@app.post("/ramalan")
async def image( request: Request):

    body = await request.form()
    if body:
      print(body['image'].file)
      return {"filename": "image.filename"}
    else :
      raise HTTPException(status_code=404, detail="error")