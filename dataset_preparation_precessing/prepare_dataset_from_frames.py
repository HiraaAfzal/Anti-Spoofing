import os

import cv2 as cv
import mediapipe as mp

# ignore warnings
import warnings
warnings.filterwarnings("ignore")


# write a function to read images from a folder, detect faces in each image using mediapipe, if a face is detected,
# crop it and save it in another folder with a counter
def prepare_dataset_from_frames(frames_path, faces_path, counter=1, face_detection_confidence=0.7):
    # Create a folder to store the faces
    if not os.path.exists(faces_path):
        os.makedirs(faces_path)

    # initialize the face detector
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=face_detection_confidence)

    # get list of frames in frames folder with full path
    frames_list = [frames_path + "\\" + frame for frame in os.listdir(frames_path)]

    # loop over the frames
    for frame in frames_list:
        # read the frame
        image = cv.imread(frame)
        # check if image is empty then skip the frame
        if image is None:
            continue
        # convert the frame to grayscale
        gray_frame = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        h, w = gray_frame.shape

        # detect the faces in the frame
        results = face_detection.process(cv.cvtColor(gray_frame, cv.COLOR_BGR2RGB))

        # check if a face is detected
        if results.detections:
            # check if more than one face is detected in the frame then skip the frame
            # if len(results.detections) > 1:
            #     continue

            detection = results.detections[0]

            # get the bounding box coordinates
            bbox = detection.location_data.relative_bounding_box
            bbox = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
            x, y, w, h = bbox

            # extract the face from the frame
            face = image[y:y + h, x:x + w]
            # check if the face is empty then skip the frame
            if face.size == 0:
                continue

            # resize the face ROI to 224x224
            face = cv.resize(face, (224, 224))

            # save the face in the faces folder
            cv.imwrite(faces_path + '\\' + str(counter) + '.jpg', face)

            # increase the counter
            counter += 1

    return counter


if __name__ == '__main__':
    # name of custom dataset folder
    custom_dataset_path = r'C:\Users\adnan\PycharmProjects\SIV-Project-AntiSpoofing\dataset_preparation_precessing\custom_dataset'

    # frames folder name
    frames_folder = 'print_attack_test'

    # path to frames folder
    frames_folder_path = custom_dataset_path + '\\' + frames_folder

    # faces folder name
    faces_folder = frames_folder + '_faces'

    # path to faces folder
    faces_folder_path = custom_dataset_path + '\\' + faces_folder

    # call the function
    print("Extracting faces...")
    counter_ = prepare_dataset_from_frames(frames_folder_path, faces_folder_path, face_detection_confidence=0.8)
    print("Total faces extracted: ", counter_ - 1)
