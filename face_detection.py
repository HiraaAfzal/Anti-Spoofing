import cv2
import mediapipe as mp
import numpy as np


class FaceDetection:
    """
    This class is used to detect faces in the image.
    """
    def __init__(self):
        """
        This function initializes the face detection model using mediapipe.
        """
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    def check_and_detect(self, frame):
        """
        This function checks if the face is present in the specified area of the frame and returns the cropped face if
        present. It also returns the bounding box of the face in the frame and the bounding box of the face in the box.
        It also checks if multiple faces are present in the frame.
        :param frame: The frame in which the face is to be detected.
        :return: A dictionary containing following keys:
        'frame': original frame with the box drawn on it.
        'box': the box in which the face is to be detected.
        'detectable': a boolean value indicating if the face is present in the box or not.
        'cropped_face': the cropped face if present.
        'box_face_bbox': the bounding box of the face according to the box.
        'frame_face_bbox': the bounding box of the face according to the frame.
        """

        face_found = False
        is_multiple = False
        detectable = False

        text = 'Scan Here'
        t_color = (230, 91, 118)
        r_color = (224, 225, 225)
        img = frame.copy()
        box = img[101:101 + 300, 179:179 + 300]

        # convert the frame to grayscale
        gray_box = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY)

        # detect the faces in the frame
        results = self.face_detection.process(cv2.cvtColor(gray_box, cv2.COLOR_BGR2RGB))

        face_crop = None
        box_bbox = None
        face_bbox_original_frame = None

        if results.detections:
            if len(results.detections) > 1:
                is_multiple = True
                detectable = False
            else:
                h, w = gray_box.shape

                # get time difference in seconds between current time and initial time
                detection = results.detections[0]

                # get the bounding box coordinates
                bbox = detection.location_data.relative_bounding_box
                bbox = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
                x, y, w, h = bbox

                box_bbox = bbox
                face_bbox_original_frame = (x + 179, y + 101), (x + w + 179, y + h + 101)

                # extract the face from the frame
                face_crop = box[y:y + h, x:x + w]

                face_found = True
                detectable = True

        # drawing sna box
        shapes = np.zeros_like(img, np.uint8)

        # Draw shapes
        cv2.rectangle(shapes, (180, 100), (450, 400), (254, 255, 255), cv2.FILLED)

        # Generate output by blending image with shapes image, using the shapes
        # images also as mask to limit the blending to those parts
        alpha = 0.88
        mask = shapes.astype(bool)
        frame[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]

        # scan box
        cv2.rectangle(frame, (180, 100), (450, 400), (254, 255, 255), 2)

        # text box
        cv2.rectangle(frame, (180, 60), (450, 95), r_color, cv2.FILLED)
        frame[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]
        cv2.putText(frame, text, (220, 90), 2, 1, t_color, 2)

        if face_found:
            detectable = True
            text = 'Face Found'
            r_color = (0, 255, 0)
            cv2.rectangle(frame, (180, 60), (450, 95), r_color, cv2.FILLED)
            cv2.putText(frame, text, (220, 90), 2, 1, t_color, 2)
        elif is_multiple:
            text = 'Multi Faces'
            r_color = (0, 0, 255)
            cv2.rectangle(frame, (180, 60), (450, 95), r_color, cv2.FILLED)
            cv2.putText(frame, text, (220, 90), 2, 1, t_color, 2)
            detectable = False

        output = {'frame': frame, 'box': box, 'detectable': detectable}

        if detectable:
            output['cropped_face'] = face_crop
            output['box_face_bbox'] = box_bbox
            output['frame_face_bbox'] = face_bbox_original_frame

        return output
