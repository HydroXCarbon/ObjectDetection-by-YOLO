from ultralytics import YOLO
import cv2
import cvzone
import math

classNames = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"
              , "trafficlight", "firehydrant", "stopsign", "parkingmeter", "bench", "bird", "cat", "dog"
              , "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella"
              , "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sportsball", "kite"
              , "baseballbat", "baseballglove", "skateboard", "surfboard", "tennisracket", "bottle"
              , "wineglass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich"
              , "orange", "broccoli", "carrot", "hotdog", "pizza", "donut", "cake", "chair", "sofa"
              , "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote"
              , "keyboard", "cellphone", "microwave", "oven", "toaster", "sink", "refrigerator", "book"
              , "clock", "vase", "scissors", "teddy bear", "hairdrier", "toothbrush", ]

# Open the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Open the video file
#cap = cv2.VideoCapture("../Videos/bikes.mp4")

model = YOLO('../Yolo-Weights/yolov8l.pt')

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:

            # Create a bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            bbox = [x1, y1, x2-x1, y2-y1]
            cvzone.cornerRect(img, bbox, l=15)

            # Create a label showing the class and confidence
            conf = math.ceil((box.conf[0] * 100))/100
            cls = int(box.cls[0])

            cvzone.putTextRect(img, f'{classNames[cls]} {conf}',(max(50,x1), max(50,y1)), scale=0.8, thickness=1)

    cv2.imshow("Image", img)
    cv2.waitKey(1)