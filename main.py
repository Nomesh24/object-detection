import cv2
import numpy as np
# import matplotlib.pyplot as plt

# yolo = cv2.dnn.readNet("./config/yolov3-tiny.weights", "./config/yolov3-tiny.cfg")


#Load yolo
net = cv2.dnn.readNet("./configv4/yolov4.weights", "./configv4/yolov4.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
classess = []
# Load class names
with open("./configv4/coco.names", 'r')as f:
  classes = f.read().splitlines()

# Initialize video capture
vid = cv2.VideoCapture(0)
if not vid.isOpened():
    print("Error: Could not open video.")
    exit()

# Get output layer names
layer_names = net.getLayerNames()
try:
    output_layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
except IndexError:
    output_layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


# output_layer_names = net.getUnconnectedOutLayersNames()
# ln = net.forward(output_layer_names)
# ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))



# function to draw bounding box on the detected object with class name
def draw_bounding_box(img, class_id, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])
    color = COLORS[class_id]
    box_thickness = 4  
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, box_thickness)
    font_scale = 1.0  # Change this value to make the text larger
    text_thickness = 3  # Change this value to make the text bolder
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, text_thickness)
  
while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    if ret:
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layer_names)

        boxes = []
        confidences = []
        classIDs = []
        h, w = frame.shape[:2]

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > 0.5:
                    box = detection[:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    box = [x, y, int(width), int(height)]
                    boxes.append(box)
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        if len(indices) > 0:
            for i in indices.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


        cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

