import cv2
import time

import joblib
import mediapipe as mp
import numpy as np


from DataTools import FullDataLabels
from GestureDenseNet import GestureDenseNet


def liveTest(model):
    mpHolistic = mp.solutions.holistic
    holistic_model = mpHolistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Initializing the drawing utils for drawing the facial landmarks on image
    mpDrawing = mp.solutions.drawing_utils

    # (0) in VideoCapture is used to connect to your computer's default camera
    capture = cv2.VideoCapture(0)

    # Initializing current time and precious time for calculating the FPS
    previousTime = 0
    currentTime = 0
    rightHandClass = ""
    leftHandClass = ""
    while capture.isOpened():
        # capture frame by frame
        ret, frame = capture.read()

        # resizing the frame for better view
        frame = cv2.resize(frame, (800, 600))

        # Converting the from BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Making predictions using holistic model
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = holistic_model.process(image)
        image.flags.writeable = True

        # Converting back the RGB image to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.right_hand_landmarks:
            rightHandLandmarks = landmarksToArray([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark])
            rightHandClass = model.predict([rightHandLandmarks])[0]

        if results.left_hand_landmarks:
            leftHandLandmarks = landmarksToArray([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark])
            leftHandClass = model.predict([leftHandLandmarks])[0]

        # Drawing Right hand Land Marks
        mpDrawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mpHolistic.HAND_CONNECTIONS
        )

        # Drawing Left hand Land Marks
        mpDrawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mpHolistic.HAND_CONNECTIONS
        )

        # Calculating the FPS
        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime

        # Displaying FPS on the image
        cv2.putText(image, str(int(fps)) + " FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f"Right Hand: {rightHandClass}", (10, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f"Left Hand: {leftHandClass}", (10, 130), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        # Display the resulting image
        cv2.imshow("Facial and Hand Landmarks", image)

        # Enter key 'q' to break the loop
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # When all the process is done
    # Release the capture and destroy all windows
    capture.release()
    cv2.destroyAllWindows()

def landmarksToArray(landmarks):
    flat_landmarks = []
    for x, y, z in landmarks:
        flat_landmarks.append(x)
        flat_landmarks.append(y)
    return flat_landmarks

if __name__ == '__main__':
    if(input("\nEnter 'L' to load in a pre-trained model from a file. Press enter to skip and train a new model\n") == 'L'):
        filename = input("\nEnter the filepath for your saved model. Leave empty for default best model.\n")
        if(filename == ""):
            model = joblib.load("models/BEST_GestureDenseNet_AllGestures_100Epochs.model")
        else:
            model = joblib.load(filename)
        print("Starting Stream")
        liveTest(model)

    else:

        d1, l1, d2, l2 = FullDataLabels()
        model = GestureDenseNet()

        print("Starting Training")
        model.train(trainingData=np.array(d1) , trainingLabels=np.array(l1), epochs=10)

        if(input("\nEnter 'S' to save your model. To skip, press enter.\n ") == 'S'):
            modelName = input("\nEnter the model name to save your model, no spaces.\n")
            joblib.dump(model, f"models/{modelName}.model")
            print(f"Saved to models/{modelName}.model")

        print("Beginning Testing")
        model.test(np.array(d2), np.array(l2))

        #Live Test
        print("Starting Stream. Press q to quit")
        liveTest(model)