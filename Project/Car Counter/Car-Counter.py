from ultralytics import YOLO
import cv2
import cvzone
import math
import torch
from sort import *

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

# Open the video file
cap = cv2.VideoCapture("../Videos/cars.mp4")
mask = cv2.imread("../Car Counter/mask.png")

# Set up the model
model = YOLO('../Yolo-Weights/yolov8l.pt')

# Set up the tracker
tracker = Sort(max_age=20 , min_hits=3, iou_threshold=0.3)

# Force the model to use the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

device_name = str(next(model.parameters()).device)
if "cuda" in device_name:
    device_name = torch.cuda.get_device_name(0)
    print(f"The model is using GPU: {device_name}")
else:
    print("The model is using CPU")

# Main loop
while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)
    results = model(imgRegion, stream=True)
    detection = np.empty((0, 5))
    for r in results:
        boxes = r.boxes
        for box in boxes:

            # Create a bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            bbox = [x1, y1, x2-x1 , y2-y1]

            # Create a label showing the class and confidence
            conf = math.ceil((box.conf[0] * 100))/100
            cls = int(box.cls[0])
            current_class = classNames[cls]

            # Only track cars, trucks, buses and motorcycles
            if (current_class == "car" or current_class == "truck" or current_class == "bus"
                    or current_class == "motorcycle" and conf > 0.3):
                cvzone.cornerRect(imgRegion, bbox, l=9, rt =5)
                cvzone.putTextRect(imgRegion, f'{current_class} {conf}', (max(50, x1), max(50, y1)), scale=0.6, thickness=1)
                currentArray = np.array([[x1, y1, x2, y2, conf]])
                deetection = np.vstack((detection, currentArray))

    result_tracker = tracker.update(detection)

    for result in result_tracker:
        x1, y1, x2, y2, id = result
        cvzone.cornerRect(imgRegion, [x1, y1, x2-x1, y2-y1], l=9, rt=5)

    cv2.imshow("imgRegion", imgRegion)
    cv2.waitKey(1)