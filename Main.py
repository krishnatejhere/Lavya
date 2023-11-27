import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import cv2
import math
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from PIL import Image

EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

def create_new_shape(coordinates):
    if len(coordinates) < 17:
        print("Invalid number of coordinates provided.")
        return None

    a = coordinates[0]
    b = ((coordinates[5][0] + coordinates[6][0]) // 2, (coordinates[5][1] + coordinates[6][1]) // 2)
    c = coordinates[5]
    d = coordinates[6]
    e = coordinates[7]
    f = coordinates[9]
    g = coordinates[8]
    h = coordinates[10]
    i = ((coordinates[11][0] + coordinates[12][0]) // 2, (coordinates[11][1] + coordinates[12][1]) // 2)
    j = coordinates[11]
    k = coordinates[13]
    l = coordinates[15]
    m = coordinates[12]
    n = coordinates[14]
    o = coordinates[16]

    new_edges = {
        (a, b): 'm',
        (a, c): 'c',
        (b, d): 'm',
        (c, e): 'c',
        (d, f): 'm',
        (e, g): 'm',
        (f, h): 'c',
        (g, i): 'c',
        (h, i): 'y',
        (i, j): 'm',
        (j, k): 'c',
        (j, l): 'y',
        (k, m): 'm',
        (l, n): 'm',
        (m, o): 'c',
        (n, o): 'c'
    }

    return new_edges


def import_image():
    Tk().withdraw()  # Hide the Tkinter root window

    # Open file dialog with image file type filters
    file_path = askopenfilename(filetypes=[('Image Files', ('*.jpeg', '*.jpg', '*.png', '*.gif'))])

    # Check if a file was selected
    if file_path:
        # Perform import operations with the selected image file
        img = Image.open(file_path)
        img.show()

        print(f"Imported image: {file_path}")
        print(f"Image size: {img.size}")

        # Convert the image to a NumPy array
        image_array = np.array(img)

        # Get the dimensions of the webcam frames
        cap = cv2.VideoCapture(0)
        _, frame = cap.read()
        resized_height, resized_width, _ = frame.shape
        cap.release()

        resized_image = cv2.resize(image_array, (resized_width, resized_height))

        # Preprocess the image if necessary
        # ... (add your preprocessing steps here)

        return resized_image

    else:
        print("No image selected.")
        return None


interpreter = tf.lite.Interpreter(model_path="D:\krishna\codes\pythonProject4\Scripts\lite-model_movenet_singlepose_thunder_3.tflite")
interpreter.allocate_tensors()


def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    ref=[]
    co=()
    po = []
    for i,kp in enumerate(shaped):
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            co=(int(kx),int(ky))
            cv2.circle(frame, (int(kx), int(ky)), 4, (255,0,0), -1)
            ref.append(co)
    return ref



def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)



def take_picture():
    # Open the webcam
    cap = cv2.VideoCapture(0)

    image = None  # Variable to store the captured image

    while True:
        # Read frame from the webcam
        ret, frame = cap.read()

        # Display the frame
        cv2.imshow('Webcam', frame)

        # Check for 'c' key press
        if cv2.waitKey(1) & 0xFF == ord('c'):
            # Assign the captured frame to the image variable
            image = frame
            print("Picture captured!")
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()

    return image

# Call the function to start capturing 
ch = input("Select import or capture (i/c): ")
if ch == 'i':
    captured_image = import_image() #Selecting image from computer
elif ch == 'c':
    captured_image = take_picture() #Selecting image from webcam
else:
    captured_image = cv2.imread(r"C:\Users\sandh\jumping jack\jumping jack_step1.jpg")
img = captured_image.copy()
img =tf.image.resize_with_pad(np.expand_dims(img, axis=0),256,256)
input_image = tf.cast(img, dtype=tf.float32)

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Make predictions
interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
interpreter.invoke()
keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
refp = draw_keypoints(captured_image, keypoints_with_scores, 0.4)
draw_connections(captured_image, keypoints_with_scores, create_new_shape(refp), 0.4)
cv2.imshow('reference pose', captured_image)
print(refp)

# Circle parameters
circle_color = (255, 255, 255)  # white
circle_radius = 5
circle_pos = (400, 300)
circle_thickness = 2

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    # Reshape
    img = frame.copy()
    img =tf.image.resize_with_pad(np.expand_dims(img, axis=0),256,256)
    input_image = tf.cast(img, dtype=tf.float32)

    # Setup input and output
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Displaying keypoints of image
    for cx,cy in refp:
        cv2.circle(frame, (cx, cy), circle_radius, (255, 255, 255), circle_thickness)

    # Make predictions
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

    # Rendering
    draw_connections(frame, keypoints_with_scores, create_new_shape(refp), 0.4)

    livp = draw_keypoints(frame, keypoints_with_scores, 0.4)

    circle_colors = []

    # Checking if points met
    for (cx, cy), (lx, ly) in zip(refp,livp):
        distance = math.sqrt((lx - cx) ** 2 + (ly - cy) ** 2)
        if distance - circle_radius < 15:
            circle_color = (0, 255, 0)  # green
            circle_radius = 7
            circle_thickness = -1
        else:
            circle_color = (0, 0, 255)  # red
            circle_radius = 5
            circle_thickness = 2
        circle_colors.append(circle_color)
        cv2.circle(frame, (cx, cy), circle_radius, circle_color, circle_thickness)

    # Check if all circle colors are green
    if all(circle_color == (0, 255, 0) for circle_color in circle_colors):
        text = "You did it!!"  # The text to be added to the image
        position = (50, 50)  # The position of the text on the image (x, y)
        font = cv2.FONT_HERSHEY_SIMPLEX  # The font used for the text
        font_scale = 1  # The size of the font
        color = (0, 0, 255)  # The color of the text in BGR format (blue, green, red)
        thickness = 2  # The thickness of the text
        cv2.putText(frame, text, position, font, font_scale, color, thickness)

    # Displaying the video
    cv2.imshow('Movenet Thunder', frame)

    # Breaking the video
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Releasing the camera and closing the window
cap.release()
cv2.destroyAllWindows()
