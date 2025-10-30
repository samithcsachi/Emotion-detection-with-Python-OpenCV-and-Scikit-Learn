import pickle
import cv2
import numpy as np
from utils import get_face_landmarks

emotions = ['HAPPY', 'SAD']


with open('./model', 'rb') as f:
    model = pickle.load(f)


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

 
    face_landmarks = get_face_landmarks(frame, draw=True, static_image_mode=False)

    if face_landmarks is not None and len(face_landmarks) > 0:
       
        X_input = np.array(face_landmarks).reshape(1, -1)

    
        output = model.predict(X_input)

    
        cv2.putText(
            frame,
            emotions[int(output[0])],
            (10, frame.shape[0] - 10),  
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            (0, 255, 0),
            5
        )
    else:
        
        cv2.putText(
            frame,
            "No face detected",
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    
    cv2.imshow('frame', frame)

    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
