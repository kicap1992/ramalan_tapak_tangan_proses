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
import json


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
    print(content_type)
    if(content_type != "image/jpeg" and content_type != "image/png" and content_type != "image/jpg") :
      raise HTTPException(status_code=404,  detail="Fail bukan foto")
    image_name = image.filename
    with open("temp/"+image_name, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
    time.sleep(2)
    sample_img = cv2.imread("temp/"+image_name)
    sample_img = cv2.flip(sample_img, 1)
    results = hands.process(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))
    if not results.multi_hand_landmarks:
      if os.path.exists("temp/"+image_name):
        os.remove("temp/"+image_name)
      raise HTTPException(status_code=404, detail="Foto harus ada telapak tangan")
        
    if len(results.multi_handedness) > 1:
      if os.path.exists("temp/"+image_name):
        os.remove("temp/"+image_name)
      raise HTTPException(status_code=404, detail="Hanya satu telapak tangan yang bisa diramal")

    tangan =  results.multi_handedness[0].classification[0].label
    if(tangan == 'Right'):
      if os.path.exists("temp/"+image_name):
        os.remove("temp/"+image_name)
      raise HTTPException(status_code=404, detail="Hanya Tangan Kiri Yang Bisa Diramal")

    thumb = None
    pinky = None
    for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):
      thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
      pinky = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x
    cek_tapak_tangan = thumb > pinky
    if(cek_tapak_tangan == False):
      if os.path.exists("temp/"+image_name):
        os.remove("temp/"+image_name)
      raise HTTPException(status_code=404, detail="Sila foto telapak tangan kiri anda")

    # if os.path.exists("temp/"+image_name):
    #   os.remove("temp/"+image_name)

    change_background_mp = mp.solutions.selfie_segmentation #untuk hapus background
    change_bg_segment = change_background_mp.SelfieSegmentation() #untuk hapus background

    sample_img1 = sample_img
    sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
    sample_img = change_bg_segment.process(sample_img)
    sample_img = sample_img.segmentation_mask > 0.9
    sample_img = np.dstack((sample_img,sample_img,sample_img))
    sample_img = np.where(sample_img, sample_img1, 255) 
    sample_img = cv2.resize(sample_img, (350,450), interpolation = cv2.INTER_AREA)
    shape = sample_img.shape    
    results = hands.process(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))
    image_height, image_width, _ = sample_img.shape


    result=None
    print(results.multi_hand_landmarks)
    for hand_landmarks in results.multi_hand_landmarks:
        annotated_image = sample_img.copy()
        palm_center_y = (hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y +
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y)/2.1
        palm_center_x = (hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x +
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x)/2.1
        myradius = int(image_height/4.9) 
        #annotated_image = makeCircle(annotated_image,palm_center_y,palm_center_x,myradius)
        y = int(palm_center_y * image_height)
        x = int(palm_center_x * image_width)
        circle_coordinates = (x,y)
        mask = np.zeros(sample_img.shape, dtype=np.uint8)
        cv2.circle(mask, circle_coordinates, myradius, (255,255,255), -1)
        ROI = cv2.bitwise_and(sample_img, mask)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        x,y,w,h = cv2.boundingRect(mask)
        result = ROI[y:y+h,x:x+w]
        mask = mask[y:y+h,x:x+w]
        result[mask==0] = (255,255,255)

    cv2.imwrite("temp/"+image_name+"cropped.png", result)
    cv2.waitKey(0)

    sample_img = cv2.imread("temp/"+image_name+"cropped.png")
    width = 450
    height = 450
    dim = (width, height)
    sample_img = cv2.resize(sample_img, dim, interpolation = cv2.INTER_AREA)
    # cv2.imshow("palm",image) #to view the palm in python
    # cv2.waitKey(0)
    gray = cv2.cvtColor(sample_img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,25,45,apertureSize = 3)
    # cv2.imshow("edges in palm",edges)
    # cv2.waitKey(0)
    edges = cv2.bitwise_not(edges)
    # cv2.imshow("edges in palm1",edges)
    cv2.imwrite("temp/"+image_name+"lines.png", edges)
    cv2.waitKey(0)

    TARGET_FILE = "temp/"+image_name+"lines.png"
    IMG_DIR = os.path.abspath(os.path.dirname(__file__)) + '/images/'
    IMG_SIZE = (200, 200)
    
    target_img = cv2.imread(TARGET_FILE)
    target_img = cv2.resize(target_img, IMG_SIZE)

    print('TARGET_FILE: %s' % (TARGET_FILE))

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    detector = cv2.AKAZE_create()
    (target_kp, target_des) = detector.detectAndCompute(target_img, None)

    hasil_ramalan = None

    datas = None

    with open('dataset.json') as f:
      datas = json.load(f)

    pilihan = 2000
    image_ramalan = None

    files = os.listdir(IMG_DIR)
    for file in files:

      comparing_img_path = IMG_DIR + file
      try:
          comparing_img = cv2.imread(comparing_img_path, cv2.IMREAD_GRAYSCALE)
          comparing_img = cv2.resize(comparing_img, IMG_SIZE)
          (comparing_kp, comparing_des) = detector.detectAndCompute(comparing_img, None)
          matches = bf.match(target_des, comparing_des)
          dist = [m.distance for m in matches]
          ret = sum(dist) / len(dist)
      except cv2.error:
          ret = 100000

      print(file, ret)
      if(ret < pilihan):
          pilihan =ret
          image_ramalan = file

    print(image_ramalan)
    theindex = None
    for index,data in enumerate(datas):
      if(data["id"] == image_ramalan):
        theindex= index
            
    hasil_ramalan = datas[theindex]['datanya']
    if os.path.exists("temp/"):
        os.remove("temp/"+image_name+"lines.png") 
        os.remove("temp/"+image_name+"cropped.png") 
        os.remove("temp/"+image_name) 

    return {"message": hasil_ramalan}

    # return {"message": "hasil_ramalan"}


# @app.post("/ramalan")
# async def image( request: Request):

#     body = await request.form()
#     if body:
#       print(body['image'].content_type)
#       return {"filename": "image.filename"}
#     else :
#       raise HTTPException(status_code=404, detail="error")

def makeCircle(img,circle_y,circle_x,radius):
    image_height, image_width, _ = img.shape
    y = int(circle_y * image_height)
    x = int(circle_x * image_width)
    circle_coordinates = (x,y)
    color = (255, 0, 0)
    thickness = 2
    return cv2.circle(img,circle_coordinates,radius,color,thickness)

@app.post("/ramalan")
async def ramalan(request: Request):
  body = await request.form()
  if(body == False) :
    raise HTTPException(status_code=404, detail="error")
  
  if(body['image'] == None and body['image'] == '' ):
    raise HTTPException(status_code=404, detail="error")

  # print(body['image'])
  image_file = body['image'] #ini file yang akan digunakan
  image_src = "temp/"+body['image'] #ini file yang akan digunakan
  size = len(image_file)
  image_name = image_file[:size - 4]

  change_background_mp = mp.solutions.selfie_segmentation #untuk hapus background
  change_bg_segment = change_background_mp.SelfieSegmentation() #untuk hapus background

  with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5) as hands: 
    
    image = cv2.flip(cv2.imread(image_src), 1)
    sample_img = cv2.flip(cv2.imread(image_src), 1)


    image = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
    image = change_bg_segment.process(image)
    image = image.segmentation_mask > 0.9
    image = np.dstack((image,image,image))
    image = np.where(image, sample_img, 255) 
    
    shape = image.shape
    
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_height, image_width, _ = image.shape
    
    for hand_landmarks in results.multi_hand_landmarks:
      palm_center_y = (hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y +
      hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y)/2.1
      palm_center_x = (hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x +
      hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x)/2.1
      myradius = int(image_height/4.9) 
      y = int(palm_center_y * image_height)
      x = int(palm_center_x * image_width)
      circle_coordinates = (x,y)
      mask = np.zeros(image.shape, dtype=np.uint8)
      cv2.circle(mask, circle_coordinates, myradius, (255,255,255), -1)
      ROI = cv2.bitwise_and(image, mask)
      mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
      x,y,w,h = cv2.boundingRect(mask)
      result = ROI[y:y+h,x:x+w]
      mask = mask[y:y+h,x:x+w]
      result[mask==0] = (255,255,255)
      cv2.imwrite("temp/"+image_name+"cropped.png", result)
      cv2.waitKey(0)

  image = cv2.imread("temp/"+image_name+"cropped.png")
  width = 450
  height = 450
  dim = (width, height)
  image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
  # cv2.imshow("palm",image) #to view the palm in python
  # cv2.waitKey(0)
  gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  edges = cv2.Canny(gray,25,45,apertureSize = 3)
  # cv2.imshow("edges in palm",edges)
  # cv2.waitKey(0)
  edges = cv2.bitwise_not(edges)
  # cv2.imshow("edges in palm1",edges)
  cv2.imwrite("temp/"+image_name+"lines.png", edges)
  cv2.waitKey(0)

  TARGET_FILE = "temp/"+image_name+"lines.png"
  IMG_DIR = os.path.abspath(os.path.dirname(__file__)) + '/images/'
  IMG_SIZE = (200, 200)
  
  target_img = cv2.imread(TARGET_FILE)
  target_img = cv2.resize(target_img, IMG_SIZE)

  print('TARGET_FILE: %s' % (TARGET_FILE))

  bf = cv2.BFMatcher(cv2.NORM_HAMMING)
  detector = cv2.AKAZE_create()
  (target_kp, target_des) = detector.detectAndCompute(target_img, None)

  hasil_ramalan = None

  datas = None

  with open('dataset.json') as f:
    datas = json.load(f)

  pilihan = 2000
  image_ramalan = None

  files = os.listdir(IMG_DIR)
  for file in files:

    comparing_img_path = IMG_DIR + file
    try:
        comparing_img = cv2.imread(comparing_img_path, cv2.IMREAD_GRAYSCALE)
        comparing_img = cv2.resize(comparing_img, IMG_SIZE)
        (comparing_kp, comparing_des) = detector.detectAndCompute(comparing_img, None)
        matches = bf.match(target_des, comparing_des)
        dist = [m.distance for m in matches]
        ret = sum(dist) / len(dist)
    except cv2.error:
        ret = 100000

    print(file, ret)
    if(ret < pilihan):
        pilihan =ret
        image_ramalan = file

  print(image_ramalan)
  theindex = None
  for index,data in enumerate(datas):
    if(data["id"] == image_ramalan):
      theindex= index
          
  hasil_ramalan = datas[theindex]['datanya']
  if os.path.exists("temp/"):
      os.remove("temp/"+image_name+"lines.png") 
      os.remove("temp/"+image_name+"cropped.png") 
      os.remove("temp/"+body['image']) 

  return {"message": hasil_ramalan}
        