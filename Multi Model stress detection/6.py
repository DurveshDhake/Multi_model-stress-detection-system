import cv2
import mediapipe as mp

# Load mediapipe face mesh model
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Function to map the number of detected eyes to a stress level (1 to 100)
def map_stress_level(num_eyes):
    return max(0, min(100, 100 - num_eyes * 25))

# Initialize mediapipe face mesh model
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    # Start video capture
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the frame as not writeable to pass by reference
        rgb_frame.flags.writeable = False

        # Process the frame
        results = face_mesh.process(rgb_frame)

        # Draw the face mesh annotations on the image
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB back to BGR
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS, 
                                          mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
                                          mp_drawing.DrawingSpec(color=(0,0,255), thickness=1, circle_radius=1))

        # Detect stress based on the presence of eyes0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml').detectMultiScale(roi_gray)
            num_eyes = len(eyes)
            stress_level = map_stress_level(num_eyes)
            if stress_level > 70:
                cv2.putText(frame, 'Stressed', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            elif stress_level > 30:
                cv2.putText(frame, 'Neutral', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            else:
                cv2.putText(frame, 'Happy', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f'Stress Level: {stress_level}', (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)

        # Display the resulting frame
        cv2.imshow('Stress Detection', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the capture
cap.release()
cv2.destroyAllWindows()

