# Face Detection using Haar Cascades with OpenCV and Matplotlib

## Aim

To write a Python program using OpenCV to perform the following image manipulations:  
i) Extract ROI from an image.  
ii) Perform face detection using Haar Cascades in static images.  
iii) Perform eye detection in images.  
iv) Perform face detection with label in real-time video from webcam.

## Software Required

- Anaconda - Python 3.7 or above  
- OpenCV library (`opencv-python`)  
- Matplotlib library (`matplotlib`)  
- Jupyter Notebook or any Python IDE (e.g., VS Code, PyCharm)

## Algorithm

### I) Load and Display Images

- Step 1: Import necessary packages: `numpy`, `cv2`, `matplotlib.pyplot`  
- Step 2: Load grayscale images using `cv2.imread()` with flag `0`  
- Step 3: Display images using `plt.imshow()` with `cmap='gray'`

### II) Load Haar Cascade Classifiers

- Step 1: Load face and eye cascade XML files 
### III) Perform Face Detection in Images

- Step 1: Define a function `detect_face()` that copies the input image  
- Step 2: Use `face_cascade.detectMultiScale()` to detect faces  
- Step 3: Draw white rectangles around detected faces with thickness 10  
- Step 4: Return the processed image with rectangles  

### IV) Perform Eye Detection in Images

- Step 1: Define a function `detect_eyes()` that copies the input image  
- Step 2: Use `eye_cascade.detectMultiScale()` to detect eyes  
- Step 3: Draw white rectangles around detected eyes with thickness 10  
- Step 4: Return the processed image with rectangles  

### V) Display Detection Results on Images

- Step 1: Call `detect_face()` or `detect_eyes()` on loaded images  
- Step 2: Use `plt.imshow()` with `cmap='gray'` to display images with detected regions highlighted  

### VI) Perform Face Detection on Real-Time Webcam Video

- Step 1: Capture video from webcam using `cv2.VideoCapture(0)`  
- Step 2: Loop to continuously read frames from webcam  
- Step 3: Apply `detect_face()` function on each frame  
- Step 4: Display the video frame with rectangles around detected faces  
- Step 5: Exit loop and close windows when ESC key (key code 27) is pressed  
- Step 6: Release video capture and destroy all OpenCV windows
## Name: C Dhanush
## Reg No: 212224040066
## Program:
```
import cv2
import matplotlib.pyplot as plt
import numpy as np
```
```
w_glass = cv2.imread('image1.png', cv2.IMREAD_GRAYSCALE)
wo_glass = cv2.imread('image2.png', cv2.IMREAD_GRAYSCALE)
group = cv2.imread('image3.png', cv2.IMREAD_GRAYSCALE)
```
```
w_glass1 = cv2.resize(w_glass, (1000, 1000))
wo_glass1 = cv2.resize(wo_glass, (1000, 1000)) 
group1 = cv2.resize(group, (1000, 1000))
```
```
plt.figure(figsize=(15,10))
plt.subplot(1,3,1);plt.imshow(w_glass1,cmap='gray');plt.title('With Glasses');plt.axis('off')
plt.subplot(1,3,2);plt.imshow(wo_glass1,cmap='gray');plt.title('Without Glasses');plt.axis('off')
plt.subplot(1,3,3);plt.imshow(group1,cmap='gray');plt.title('Group Image');plt.axis('off')
plt.show()
```
```
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def detect_and_display(image):
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 10)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()
```
```
import cv2
from matplotlib import pyplot as plt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    print("Error: Cascade file not loaded properly!")
else:
    print("Cascade loaded successfully.")
w_glass1 = cv2.imread('image1.png')  # <-- replace with your image filename

if w_glass1 is None:
    print("Error: Image not found. Check the filename or path.")
else:
    print("Image loaded successfully.")
def detect_and_display(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)
    
    return image
if w_glass1 is not None and not face_cascade.empty():
    result = detect_and_display(w_glass1)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
```
```
import cv2
from matplotlib import pyplot as plt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
if face_cascade.empty():
    print("Error: Face cascade not loaded properly!")
if eye_cascade.empty():
    print("Error: Eye cascade not loaded properly!")
# (Change the filenames as per your actual image files)
w_glass = cv2.imread('image1.png')
wo_glass = cv2.imread('image2.png')
group = cv2.imread('image3.png')
def detect_eyes(image):
    face_img = image.copy()
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(face_img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    
    return face_img
if w_glass is not None:
    w_glass_result = detect_eyes(w_glass)
    plt.imshow(cv2.cvtColor(w_glass_result, cv2.COLOR_BGR2RGB))
    plt.title("With Glasses - Eye Detection")
    plt.axis("off")
    plt.show()

if wo_glass is not None:
    wo_glass_result = detect_eyes(wo_glass)
    plt.imshow(cv2.cvtColor(wo_glass_result, cv2.COLOR_BGR2RGB))
    plt.title("Without Glasses - Eye Detection")
    plt.axis("off")
    plt.show()

if group is not None:
    group_result = detect_eyes(group)
    plt.imshow(cv2.cvtColor(group_result, cv2.COLOR_BGR2RGB))
    plt.title("Group - Eye Detection")
    plt.axis("off")
    plt.show()
```
```
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis("on")
    plt.title("Video Face Detection")
    plt.show()
    break

cap.release()
```
```
from IPython.display import clear_output, display
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
```
```
def new_detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    return frame
```
```
video_capture = cv2.VideoCapture(0)
captured_frame = None   

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("No frame captured from camera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = new_detect(gray, frame)
    clear_output(wait=True)
    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Video - Face & Eye Detection")
    display(plt.gcf())
    captured_frame = canvas.copy()  
    break
```
```
video_capture.release()
if captured_frame is not None and captured_frame.size > 0:
    cv2.imwrite('captured_face_eye.png', captured_frame)
    captured_image = cv2.imread('captured_face_eye.png', cv2.IMREAD_GRAYSCALE)
    plt.imshow(captured_image, cmap='gray')
    plt.title('Captured Face with Eye Detection')
    plt.axis('off')
    plt.show()
else:
    print("No valid frame to save.")
```
```
image = cv2.imread('image4.png') 
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis('off')

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0) 

edges = cv2.Canny(blurred_image, 50, 150)  

plt.imshow(edges, cmap='gray')
plt.title("Canny Edge Detection")
plt.axis('off')
```
```
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

result_image = image.copy() 
for contour in contours:
    if cv2.contourArea(contour) > 50: 
        x, y, w, h = cv2.boundingRect(contour)  
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  


plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
plt.title("Handwriting Detection")
plt.axis('off')
```
```
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Paths to model files
prototxt = "deploy.prototxt"
model = "mobilenet_iter_73000.caffemodel"

# Load the pre-trained MobileNet-SSD model
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# Class labels the model can detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

# Random colors for each class
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Load your image
image_path = "image5.png"   # 🖼️ Replace with your image filename
image = cv2.imread(image_path)
(h, w) = image.shape[:2]

# Prepare the image as input to the DNN
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)),
                             0.007843, (300, 300), 127.5)

# Forward pass through the network
net.setInput(blob)
detections = net.forward()

# Loop over the detections
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.5:
        idx = int(detections[0, 0, i, 1])
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        label = f"{CLASSES[idx]}: {confidence * 100:.2f}%"
        color = COLORS[idx]
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(image, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Convert BGR to RGB for Matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Show the output using Matplotlib
plt.imshow(image_rgb)
plt.title("Object Detection with MobileNet-SSD")
plt.axis("off")
plt.show()
```
## Output:

<img width="435" height="447" alt="Screenshot 2025-11-09 200315" src="https://github.com/user-attachments/assets/07292182-22c1-4872-8c1b-08eb8f0b5b57" />
<br>
<img width="459" height="452" alt="Screenshot 2025-11-09 200336" src="https://github.com/user-attachments/assets/522b34d7-9012-4799-9eac-c48be925977d" />
<br>
<img width="461" height="445" alt="Screenshot 2025-11-09 200400" src="https://github.com/user-attachments/assets/67a394fa-1ab1-409b-ae48-0d9e81c58c8e" />
<br>
<img width="498" height="517" alt="Screenshot 2025-11-09 200618" src="https://github.com/user-attachments/assets/29287b02-760d-40ce-bae5-c0f729518c16" />
<br>
<img width="367" height="506" alt="Screenshot 2025-11-09 200651" src="https://github.com/user-attachments/assets/b3500b6b-2315-4143-b27c-21ea3ea214f1" />
<br>
<img width="382" height="497" alt="Screenshot 2025-11-09 200715" src="https://github.com/user-attachments/assets/e86599aa-166e-4329-8356-93b378ca819d" />
<br>
<img width="713" height="556" alt="Screenshot 2025-11-09 200736" src="https://github.com/user-attachments/assets/1e3152ac-0821-4c77-93b9-62083ec0f716" />
<br>
<img width="626" height="510" alt="Screenshot 2025-11-09 200759" src="https://github.com/user-attachments/assets/c55732af-0a1f-4b2a-a013-ff3a145c7d7e" />
<br>
<img width="648" height="513" alt="Screenshot 2025-11-09 200827" src="https://github.com/user-attachments/assets/70e9f804-b48c-43c0-81d1-b09085573903" />
<br>
<img width="672" height="365" alt="Screenshot 2025-11-09 200903" src="https://github.com/user-attachments/assets/cc30d767-a676-41c5-a62f-6da3bc9c9848" />
<br>
<img width="690" height="371" alt="Screenshot 2025-11-09 200916" src="https://github.com/user-attachments/assets/dc3525e4-3d42-4e60-b85a-090749f442bd" />
<br>
<img width="446" height="513" alt="Screenshot 2025-11-09 202733" src="https://github.com/user-attachments/assets/efb8840c-7c4e-4efe-a5da-5a2e76029ee9" />





