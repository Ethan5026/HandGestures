import os
from boxsdk import OAuth2, Client
import cv2
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import mediapipe as mp
from concurrent.futures import ThreadPoolExecutor


# Box authentication (replace with your credentials)
def authenticate_box():
    auth = OAuth2(
        client_id='YOUR_CLIENT_ID',
        client_secret='YOUR_CLIENT_SECRET',
        access_token='YOUR_ACCESS_TOKEN',
    )
    return Client(auth)


# Initialize MediaPipe Hand Landmarker
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)


# Feature extraction function
def extract_features(image):
    # Convert image to RGB for MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 1. Hand Landmarks (MediaPipe)
    results = hands.process(image_rgb)
    if not results.multi_hand_landmarks:
        return None  # Skip if no hand detected
    landmarks = results.multi_hand_landmarks[0]
    landmark_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten()  # 21 * 3 = 63 features

    # Normalize landmarks (relative to wrist, landmark 0)
    wrist = np.array([landmarks.landmark[0].x, landmarks.landmark[0].y, landmarks.landmark[0].z])
    normalized_landmarks = landmark_array.reshape(-1, 3) - wrist
    landmark_features = normalized_landmarks.flatten()

    # 2. Hand Symmetry (simple example: distance symmetry across midline)
    mid_x = (landmarks.landmark[5].x + landmarks.landmark[17].x) / 2  # Midpoint between index and pinky base
    left_points = [landmarks.landmark[i] for i in [5, 6, 7, 8]]  # Index finger
    right_points = [landmarks.landmark[i] for i in [17, 18, 19, 20]]  # Pinky finger
    symmetry_score = np.mean([abs((lp.x - mid_x) + (rp.x - mid_x)) for lp, rp in zip(left_points, right_points)])

    # 3. Hand Texture (HOG on hand region)
    x_min, y_min = int(min(lm.x for lm in landmarks.landmark) * image.shape[1]), int(
        min(lm.y for lm in landmarks.landmark) * image.shape[0])
    x_max, y_max = int(max(lm.x for lm in landmarks.landmark) * image.shape[1]), int(
        max(lm.y for lm in landmarks.landmark) * image.shape[0])
    hand_roi = image[y_min:y_max, x_min:x_max]
    if hand_roi.size == 0:
        return None
    hand_gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)  # Resize to 64x64 for consistency
    resized_roi = cv2.resize(hand_gray, (64, 64))
    texture_features = hog.compute(resized_roi).flatten()

    # Combine all features
    combined_features = np.concatenate([
        landmark_features,  # 63 features (21 landmarks * 3)
        np.array([symmetry_score]),  # 1 feature
        texture_features  # HOG features (depends on params, ~1764 with these settings)
    ])
    return combined_features


# Process a single image
def process_image(file_object, gesture_label):
    img_data = file_object.content()
    np_img = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    features = extract_features(img)
    if features is None:
        return None, None
    return features, gesture_label


# Stream and train incrementally
def stream_and_train_model(box_client, folder_id, gesture_classes, chunk_size=100):
    model = SGDClassifier(loss='hinge', learning_rate='optimal', max_iter=1000)
    scaler = StandardScaler()
    first_chunk = True

    label_map = {gesture: idx for idx, gesture in enumerate(gesture_classes)}
    folder = box_client.folder(folder_id).get_items(limit=None)

    X_chunk = []
    y_chunk = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        for item in folder:
            if item.type == 'file':
                file_name = item.name.lower()
                gesture_label = next((g for g in gesture_classes if g in file_name), None)
                if gesture_label is None:
                    continue

                future = executor.submit(process_image, item, label_map[gesture_label])
                features, label = future.result()
                if features is None:
                    continue

                X_chunk.append(features)
                y_chunk.append(label)

                if len(X_chunk) >= chunk_size:
                    X_chunk = np.array(X_chunk)
                    y_chunk = np.array(y_chunk)

                    if first_chunk:
                        scaler.fit(X_chunk)
                        first_chunk = False
                    X_chunk_scaled = scaler.transform(X_chunk)

                    model.partial_fit(X_chunk_scaled, y_chunk, classes=np.arange(len(gesture_classes)))
                    X_chunk, y_chunk = [], []
                    print(f"Updated model with chunk of {chunk_size} images")

        if X_chunk:
            X_chunk = np.array(X_chunk)
            y_chunk = np.array(y_chunk)
            X_chunk_scaled = scaler.transform(X_chunk)
            model.partial_fit(X_chunk_scaled, y_chunk, classes=np.arange(len(gesture_classes)))
            print("Updated model with final chunk")

    return model, scaler


def main():
    gesture_classes = ['thumbs_up', 'thumbs_down', 'fist', 'open_palm', 'okay_sign']
    box_client = authenticate_box()
    folder_id = 'YOUR_FOLDER_ID'

    model, scaler = stream_and_train_model(box_client, folder_id, gesture_classes, chunk_size=100)

    import joblib
    joblib.dump(model, 'gesture_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Model training completed and saved.")


if __name__ == "__main__":
    main()