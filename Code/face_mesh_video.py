# 1. Installing and Importing Dependencies
import mediapipe as mp
import cv2
import time

mp_drawing = mp.solutions.drawing_utils  # Helpers
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh  # MediaPipe Solution

# 2. Make Detections (webcam input)
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(1)

# 3. Initializing Face Mesh Model
with mp_face_mesh.FaceMesh(max_num_faces=1,
                           refine_landmarks=True,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        ret, image = cap.read()
        start = time.time()

        # Flip the image horizontally for a later selfie-view display
        # Convert the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False

        # Process the image and make Detections
        results = face_mesh.process(image)
        image.flags.writeable = True

        # Convert the image color back to RGB for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw face landmarks
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # print(face_landmarks) # remove for better performance
                # print(face_landmarks.landmark.x)
                # Draw the face mesh annotations on the image.
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    # connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)

        end = time.time()
        totalTime = end-start
        if totalTime != 0:
            fps = 1/totalTime

        cv2.putText(image, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
        cv2.imshow('Raw Webcam Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
