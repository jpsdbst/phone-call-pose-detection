import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Load YOLOv8 model
model = YOLO("yolov8m.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Setup MediaPipe Pose
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_copy = frame.copy()

        # ---------- YOLOv8 ----------
        yolo_results = model.predict(source=frame, show=False, verbose=False)
        cell_phone_detected = False

        for result in yolo_results:
            for box in result.boxes:
                cls = int(box.cls[0])
                label = model.names[cls].lower()
                conf = float(box.conf[0])

                if 'cell' in label and conf > 0.4:  # Filter for cellphone
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame_copy, f'{label} {conf:.2f}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    cell_phone_detected = True

        # ---------- MediaPipe Pose ----------
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = pose.process(image_rgb)
        image_rgb.flags.writeable = True
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Draw pose landmarks
        arm_detected = False
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

            # Arm detection logic
            try:
                landmarks = results.pose_landmarks.landmark
                left = all(landmarks[p].visibility > 0.5 for p in [
                    mp_pose.PoseLandmark.LEFT_SHOULDER,
                    mp_pose.PoseLandmark.LEFT_ELBOW,
                    mp_pose.PoseLandmark.LEFT_WRIST
                ])
                right = all(landmarks[p].visibility > 0.5 for p in [
                    mp_pose.PoseLandmark.RIGHT_SHOULDER,
                    mp_pose.PoseLandmark.RIGHT_ELBOW,
                    mp_pose.PoseLandmark.RIGHT_WRIST
                ])
                # Final arm + cellphone check
                arm_detected = (left or right) and cell_phone_detected
            except:
                pass

        if arm_detected:
            cv2.putText(image_bgr, 'Arm + Cell Phone Detected!', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Combine YOLO detections with pose
        combined = cv2.addWeighted(image_bgr, 1, frame_copy, 0.5, 0)
        cv2.imshow('YOLO + MediaPipe', combined)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()




# import cv2
# import mediapipe as mp
# import numpy as np
# from ultralytics import YOLO

# mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose

# # Load YOLOv8 model
# model = YOLO("yolov8m.pt")

# # Open webcam
# cap = cv2.VideoCapture(0)

# # Setup MediaPipe Pose
# with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # 1. ---------- YOLOv8 ----------
#         yolo_results = model.predict(source=frame, show=False, verbose=False)
#         yolo_frame = frame.copy()
#         for result in yolo_results:
#             for box in result.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 conf = float(box.conf[0])
#                 cls = int(box.cls[0])
#                 label = model.names[cls]
#                 # Draw blue rectangle
#                 cv2.rectangle(yolo_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
#                 cv2.putText(yolo_frame, f'{label} {conf:.2f}', (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


#         # 2. ---------- MediaPipe Pose ----------
#         image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         image_rgb.flags.writeable = False
#         results = pose.process(image_rgb)
#         image_rgb.flags.writeable = True
#         image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

#         # Draw pose landmarks
#         arm_detected = False
#         if results.pose_landmarks:
#             mp_drawing.draw_landmarks(
#                 image_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#                 mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
#                 mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
#             )

#             # Arm detection logic
#             try:
#                 landmarks = results.pose_landmarks.landmark
#                 left = all(landmarks[p].visibility > 0.5 for p in [
#                     mp_pose.PoseLandmark.LEFT_SHOULDER,
#                     mp_pose.PoseLandmark.LEFT_ELBOW,
#                     mp_pose.PoseLandmark.LEFT_WRIST
#                 ])
#                 right = all(landmarks[p].visibility > 0.5 for p in [
#                     mp_pose.PoseLandmark.RIGHT_SHOULDER,
#                     mp_pose.PoseLandmark.RIGHT_ELBOW,
#                     mp_pose.PoseLandmark.RIGHT_WRIST
#                 ])
#                 arm_detected = left or right
#             except:
#                 pass

#         if arm_detected:
#             cv2.putText(image_bgr, 'Arm Detected!', (50, 50),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#         # 3. ---------- Combine Results ----------
#         # Use YOLO boxes drawn on top of MediaPipe frame
#         combined = cv2.addWeighted(image_bgr, 1, yolo_frame, 0.5, 0)

#         # Show final output
#         cv2.imshow('YOLO + MediaPipe', combined)

#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()







# import cv2
# import mediapipe as mp
# import numpy as np
# mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose

# from PIL import Image

# from ultralytics import YOLO


# model = YOLO("yolov8m.pt")
# # accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
# results = model.predict(source="0", show= True)

# cap = cv2.VideoCapture(0)
# ## Setup mediapipe instance
# with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         # Recolor image to RGB
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         image.flags.writeable = False
      
#         # Make detection
#         results = pose.process(image)

#         # Recolor back to BGR
#         image.flags.writeable = True
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#         arm_detected = False
#         try:
#             landmarks = results.pose_landmarks.landmark
#             # Check visibility of left arm landmarks
#             left_shoulder_vis = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility
#             left_elbow_vis = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility
#             left_wrist_vis = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].visibility

#             # Check visibility of right arm landmarks
#             right_shoulder_vis = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility
#             right_elbow_vis = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility
#             right_wrist_vis = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].visibility

#             # Detect if either arm is visible
#             if (left_shoulder_vis > 0.5 and left_elbow_vis > 0.5 and left_wrist_vis > 0.5) or \
#                (right_shoulder_vis > 0.5 and right_elbow_vis > 0.5 and right_wrist_vis > 0.5):
#                 arm_detected = True
#         except:
#             pass

#         # Render detections
#         mp_drawing.draw_landmarks(
#             image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
#             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
#         )

#         # Show message if arm detected
#         if arm_detected:
#             cv2.putText(image, 'Arm Detected!', (50, 50),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

#         cv2.imshow('Mediapipe Feed', image)

#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

   
#     cv2.imshow('Mediapipe Feed', image)



#     len(landmarks)
#     for lndmrk in mp_pose.PoseLandmark:
#         print(lndmrk)
#     print(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility)

    