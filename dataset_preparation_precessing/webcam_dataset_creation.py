import os
import warnings

import cv2 as cv
import mediapipe as mp

warnings.filterwarnings("ignore")


# write a function to read frames from webcam, detect faces in each frame using mediapipe, if a face is detected,
# crop it and save it in another folder with a counter variable and also create a video of actual frames as well.
def webcam_dataset_creation(frames_path, faces_path, no_of_faces=200, cam_index=0, face_detection_confidence=0.8, skip_frames=5):
    # Create a folder to store the faces
    if not os.path.exists(faces_path):
        os.makedirs(faces_path)

    # Create a folder to store the frames
    if not os.path.exists(frames_path):
        os.makedirs(frames_path)

    # initialize the face detector
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=face_detection_confidence)

    # initialize the webcam
    cam = cv.VideoCapture(cam_index)

    # initialize the counter variable
    counter = 1
    face_counter = 0

    # loop over the frames
    while True:
        if not counter % skip_frames == 0:
            counter += 1
            continue
        # read the frame
        ret, image = cam.read()
        # check if image is empty then skip the frame
        if image is None:
            continue

        # flip the frame
        image = cv.flip(image, 1)

        # copy the frame
        frame = image.copy()

        # convert the frame to grayscale
        gray_frame = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        h, w = gray_frame.shape

        # detect the faces in the frame
        results = face_detection.process(cv.cvtColor(gray_frame, cv.COLOR_BGR2RGB))

        # check if a face is detected
        if results.detections:
            # check if more than one face is detected in the frame then skip the frame
            if len(results.detections) > 1:
                continue

            detection = results.detections[0]

            # get the bounding box coordinates
            bbox = detection.location_data.relative_bounding_box
            bbox = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
            x, y, w, h = bbox

            # extract the face from the frame
            face = image[y:y + h, x:x + w]

            if face.size == 0:
                continue

            # resize the face ROI to 224x224
            face = cv.resize(face, (224, 224))

            # save the face in a folder with a counter variable
            cv.imwrite(faces_path + '/face' + str(counter) + '.jpg', face)
            # increase the counter variable
            counter += 1

            # check if the counter variable reaches the no_of_faces then break the loop
            if face_counter > no_of_faces:
                break

            face_counter += 1

            # draw the bounding box on the frame
            cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # save the frame in a folder with a counter variable
        cv.imwrite(frames_path + '/frame' + str(counter) + '.jpg', frame)

        # show the frame
        cv.imshow('image', image)

        # if the 'q' key is pressed then break the loop
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # release the webcam
    cam.release()
    # destroy all the windows
    cv.destroyAllWindows()

    # record the video of the frames in the frames_path folder
    video_path = frames_path + '.avi'
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter(video_path, fourcc, 20.0, (640, 480))
    for i in range(1, counter):
        frame = cv.imread(frames_path + '/frame' + str(i) + '.jpg')
        out.write(frame)

    # release the video writer
    out.release()

    # destroy all the windows
    cv.destroyAllWindows()

    return counter


if __name__ == '__main__':
    # path to store the frames
    frames_path = 'adnan'
    # path to store the faces
    faces_path = 'adnan_faces'
    # number of faces to be detected
    no_of_faces = 200
    # webcam index
    cam_index = 0
    # face detection confidence
    face_detection_confidence = 0.8
    skip_frames = 8

    # call the function
    webcam_dataset_creation(frames_path, faces_path, no_of_faces, cam_index, face_detection_confidence, skip_frames)
