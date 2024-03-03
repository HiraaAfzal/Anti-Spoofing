import time
import os
import cv2
import mediapipe as mp
import numpy as np

from face_detection import FaceDetection
from face_liveness_detection import FaceLiveness_EyeMouthMovement, FaceLiveness_Colorspace_Print_Replay_Attack

import warnings
# ignore all future warnings from sklearn
warnings.simplefilter(action='ignore', category=FutureWarning)
# ignore all warnings from sklearn
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")


CAM_INDEX = 0
LIVENESS_CLASSIFIER_PATH = os.path.join(os.getcwd(), 'Models', 'model.pkl')

T_COLOR = (230, 91, 118)
R_COLOR = (224, 225, 225)
NO_OF_FRAMES_TO_CHECK_FOR_COLOR_SPACE = 10

# time to check for eye and mouth movement in seconds after the face is detected in the frame
TIME_CHECK_FOR_EYE_MOUTH_MOVEMENT = 10

# time to check for space in seconds after the face is detected in the frame
TIME_CHECK_FOR_COLOR_SPACE = 10

# time check to reset the liveness check after liveness is detected
TIME_CHECK_TO_RESET_LIVENESS = 10


# method to initialize the webcam
def cam_init(cam_index, width, height):
    """
    Initialize the camera.
    :param cam_index: Camera index to be used. 0 for default camera.
    :param width: width of the frame to be captured
    :param height: height of the frame to be captured
    :return: VideoCapture object
    """
    cap_ = cv2.VideoCapture(cam_index)

    cap_.set(cv2.CAP_PROP_FPS, 30)
    cap_.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap_.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    return cap_


def transform_face_landmarks(face_landmarks_):
    """
    Transform face landmarks into pixel coordinates(list of tuples) for drawing on the image.
    :param face_landmarks_: face landmarks from mediapipe in normalized coordinates
    :return: list of tuples of pixel coordinates
    """
    # transform the face landmarks into pixel coordinates
    co_ordinates = [(int(landmark_point.x * w), int(landmark_point.y * h))
                    for landmark_point in face_landmarks_.multi_face_landmarks[0].landmark]

    return co_ordinates


if __name__ == '__main__':
    # initialize the camera
    cap = cam_init(CAM_INDEX, 640, 480)

    # initialize FaceLiveness_EyeMouthMovement class
    liveness_movement_detector = FaceLiveness_EyeMouthMovement()

    # initialize FaceLiveness_Colorspace_Print_Replay_Attack class
    liveness_colorspace_detector = FaceLiveness_Colorspace_Print_Replay_Attack(LIVENESS_CLASSIFIER_PATH)

    # initialize the mediapipe drawing utilities
    mp_drawing = mp.solutions.drawing_utils

    # initialize the face landmark detector
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # initialize the face detector
    face_detector = FaceDetection()

    liveness_by_eye_mouth_movement = False
    liveness_by_colorspace = False
    reset_flag = False

    # list to store color_space predictions for 'NO_OF_FRAMES_TO_CHECK_FOR_COLOR_SPACE'
    color_space_predictions = []
    # Counter to keep track of the number of frames for which the color_space prediction is made
    color_space_frame_counter = 0

    # get the initial time in seconds
    initial_time = time.time()
    while True:
        # read the frame from the camera
        ret, frame = cap.read()

        # flip the frame
        frame = cv2.flip(frame, 1)

        # convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect the faces in the frame
        face_detection_results = face_detector.check_and_detect(frame)

        if face_detection_results['detectable']:
            # detect face from the original frame
            results = face_detector.face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.detections:
                if len(results.detections) > 1:
                    # if more than one face is detected in the frame, then skip the frame
                    continue

                # get the first detection
                detection = results.detections[0]

                # get the height and width of the frame
                h, w = gray_frame.shape

                # get current time in seconds
                current_time = time.time()

                # get time difference in seconds between current time and initial time
                time_diff = current_time - initial_time

                # get the bounding box of the face
                bbox = detection.location_data.relative_bounding_box
                bbox = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
                x, y, w, h = bbox

                # draw the bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # crop the face from the frame
                face = frame[y:y + h, x:x + w]

                # if size of face is empty, continue
                if face.size == 0:
                    continue

                # convert the face to grayscale
                gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

                # detect the face landmarks on the face
                results = face_mesh.process(cv2.cvtColor(gray_face, cv2.COLOR_BGR2RGB))

                # check if the face landmarks are detected
                if results.multi_face_landmarks:
                    # transform the face landmarks into pixel coordinates
                    landmarks_coords = transform_face_landmarks(results)

                    # draw the eyes shape on the frame
                    cv2.polylines(face, [np.array([landmarks_coords[p] for p in liveness_movement_detector.LEFT_EYE],
                                                  dtype=np.int32)], True, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.polylines(face, [np.array([landmarks_coords[p] for p in liveness_movement_detector.RIGHT_EYE],
                                                  dtype=np.int32)], True, (0, 255, 0), 1, cv2.LINE_AA)

                    # draw the lips shape on the frame
                    cv2.polylines(face, [np.array([landmarks_coords[p] for p in liveness_movement_detector.LIPS],
                                                  dtype=np.int32)], True, (0, 255, 0), 1, cv2.LINE_AA)

                    if liveness_by_eye_mouth_movement is False and reset_flag is False:
                        # show liveness message
                        cv2.putText(frame, "Blink or Speak for Liveness...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 255, 0), 2)

                        # show the timer in the top left corner of the frame
                        height_, width_ = frame.shape[:2]
                        timer_position = (width_ - 100, 30)
                        cv2.putText(frame, "Timer: " + str(int(TIME_CHECK_FOR_EYE_MOUTH_MOVEMENT - time_diff)),
                                    timer_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        # check if the face is alive or not using eye and mouth movement detection
                        is_face_live_by_movement = liveness_movement_detector.detect_liveness_by_eye_mouth_movement(
                            landmarks_coords)

                        # if time difference is greater than time_check_for_eye_mouth_movement seconds
                        if time_diff > TIME_CHECK_FOR_EYE_MOUTH_MOVEMENT and is_face_live_by_movement is False:
                            # reset the initial time
                            initial_time = time.time()
                            message = "Spoof Detected"
                            reset_flag = True
                            liveness_by_colorspace = False
                        else:
                            if is_face_live_by_movement and liveness_by_colorspace is False:
                                liveness_by_eye_mouth_movement = True
                                reset_flag = False

                    if liveness_by_eye_mouth_movement is True and reset_flag is False:
                        #######################################################################################
                        # Now check if the face is real or not using color space
                        #######################################################################################
                        # resize the face to 224x224
                        face = cv2.resize(face, (224, 224))

                        # predict the liveness of the face using color and texture
                        max_prob, max_prob_class = liveness_colorspace_detector.predict_liveness([face])

                        color_space_predictions.append((max_prob, max_prob_class))
                        color_space_frame_counter += 1

                        if NO_OF_FRAMES_TO_CHECK_FOR_COLOR_SPACE % 2 == 0:
                            if color_space_frame_counter == NO_OF_FRAMES_TO_CHECK_FOR_COLOR_SPACE + 1:
                                # get the most frequent class
                                max_prob_class = max(set(color_space_predictions),
                                                     key=color_space_predictions.count)
                                # get the most frequent probability
                                max_prob = max(color_space_predictions, key=color_space_predictions.count)
                                liveness_by_colorspace = True
                                # reset the initial time
                                initial_time = time.time()
                                reset_flag = True

                                # print(max_prob_class[1], ":", max_prob[0])
                        else:
                            if color_space_frame_counter == NO_OF_FRAMES_TO_CHECK_FOR_COLOR_SPACE:
                                # get the most frequent class
                                max_prob_class = max(set(color_space_predictions),
                                                     key=color_space_predictions.count)

                                # get the most frequent probability
                                max_prob = max(color_space_predictions, key=color_space_predictions.count)
                                liveness_by_colorspace = True
                                liveness_by_eye_mouth_movement = True

                                # reset the initial time
                                initial_time = time.time()
                                reset_flag = True

                                # print(max_prob_class[1], ":", max_prob[0])

                    if liveness_by_eye_mouth_movement is False and liveness_by_colorspace is False and reset_flag is True:
                        # get current time in seconds
                        current_time = time.time()

                        # get time difference in seconds between current time and initial time
                        time_diff = current_time - initial_time
                        if time_diff > TIME_CHECK_TO_RESET_LIVENESS:
                            # reset the initial time
                            initial_time = time.time()
                            reset_flag = False

                        R_COLOR = (0, 0, 255)
                        cv2.rectangle(frame, (180, 60), (450, 95), R_COLOR, cv2.FILLED)
                        cv2.putText(frame, message, (195, 87), 2, 1, T_COLOR, 2)

                    if liveness_by_eye_mouth_movement is True and liveness_by_colorspace is True and reset_flag is True:
                        # get current time in seconds
                        current_time = time.time()

                        # get time difference in seconds between current time and initial time
                        time_diff = current_time - initial_time
                        if time_diff > TIME_CHECK_TO_RESET_LIVENESS:
                            # reset the initial time
                            initial_time = time.time()
                            reset_flag = False
                            liveness_by_eye_mouth_movement = False
                            liveness_by_colorspace = False
                            color_space_predictions = []
                            color_space_frame_counter = 0

                        if str(max_prob_class[1]).lower() == 'live':
                            G_COLOR = (0, 255, 0)
                            cv2.rectangle(frame, (180, 60), (450, 95), G_COLOR, cv2.FILLED)
                            cv2.putText(frame, "Face is Live", (220, 87), 2, 1, T_COLOR, 2)
                        else:
                            R_COLOR = (0, 0, 255)
                            cv2.rectangle(frame, (180, 60), (450, 95), R_COLOR, cv2.FILLED)
                            # cv2.putText(frame, str(max_prob_class[1]), (220, 87), 2, 1, T_COLOR, 2)
                            cv2.putText(frame, "Spoof Detected", (200, 87), 2, 1, T_COLOR, 2)

        # display the frame
        cv2.imshow('Face Anti-Spoofing', frame)

        # press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release the camera
    cap.release()
    cv2.destroyAllWindows()
