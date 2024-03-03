import cv2
import numpy as np
import joblib


def euclidean_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points by using the formula:
    sqrt((x2 - x1)^2 + (y2 - y1)^2)
    :param point1:
    :param point2:
    :return: euclidean distance
    """
    x1_, y1_ = point1
    x2, y2 = point2
    distance = np.sqrt((x2 - x1_) ** 2 + (y2 - y1_) ** 2)
    return distance


class FaceLiveness_EyeMouthMovement:
    """
    This class is used to detect the liveness of the face by using the eye and mouth movement.
    """
    def __init__(self, eye_threshold=4.5, mouth_threshold=2.0, closed_eyes_frame=2, opened_mouth_frame=2):
        """
        :param eye_threshold: by default 4.5 for the eye aspect ratio
        :param mouth_threshold: by default 2.0 for the mouth aspect ratio
        :param closed_eyes_frame: by default 2 for the number of frames that the eyes are closed
        :param opened_mouth_frame: by default 2 for the number of frames that the mouth is opened
        """
        self.eye_threshold = eye_threshold
        self.mouth_threshold = mouth_threshold

        self.CEF_COUNTER = 0
        self.TOTAL_BLINKS = 0
        self.CLOSED_EYES_FRAME = closed_eyes_frame

        self.CMF_COUNTER = 0
        self.TOTAL_MOUTH_OPEN = 0
        self.OPENED_MOUTH_FRAME = opened_mouth_frame

        # lips indices for Landmarks
        self.LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
                     185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
        self.LOWER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88,
                           95]
        self.UPPER_LIPS = [185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
        # Left eyes indices
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        # right eyes indices
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

    @staticmethod
    def blink_ratio(landmarks, right_indices, left_indices):
        """
        Calculate the eye aspect ratio by calculating the euclidean distance between the horizontal and vertical
        landmarks of the  eyes and then dividing the horizontal distance by the vertical distance.
        Finally adding the ratio of the left eye and the right eye and dividing by 2 to get the average ratio.
        :param landmarks: landmark pixels coordinates extracted from the face
        :param right_indices: right eye indices list
        :param left_indices: left eye indices list
        :return: eye aspect ratio
        """
        # RIGHT_EYE
        # horizontal line
        rh_right = landmarks[right_indices[0]]
        rh_left = landmarks[right_indices[8]]

        # vertical line
        rv_top = landmarks[right_indices[12]]
        rv_bottom = landmarks[right_indices[4]]

        # LEFT_EYE
        # horizontal line
        lh_right = landmarks[left_indices[0]]
        lh_left = landmarks[left_indices[8]]

        # vertical line
        lv_top = landmarks[left_indices[12]]
        lv_bottom = landmarks[left_indices[4]]

        rh_distance = euclidean_distance(rh_right, rh_left)
        rv_distance = euclidean_distance(rv_top, rv_bottom)

        lv_distance = euclidean_distance(lv_top, lv_bottom)
        lh_distance = euclidean_distance(lh_right, lh_left)

        re_ratio = rh_distance / rv_distance
        le_ratio = lh_distance / lv_distance

        ratio_ = (re_ratio + le_ratio) / 2
        return ratio_

    @staticmethod
    def mouth_ratio(landmarks, lower_lip_indices, upper_lip_indices):
        """
        Calculate the mouth aspect ratio by calculating the euclidean distance between the horizontal and vertical
        landmarks of the mouth and then dividing the horizontal distance by the vertical distance.
        :param landmarks: landmark pixels coordinates extracted from the face
        :param lower_lip_indices: lower lip indices list
        :param upper_lip_indices: upper lip indices list
        :return: mouth aspect ratio
        """
        # horizontal line
        rh_right = landmarks[lower_lip_indices[0]]
        rh_left = landmarks[lower_lip_indices[8]]
        # vertical line
        rv_top = landmarks[upper_lip_indices[6]]
        rv_bottom = landmarks[lower_lip_indices[6]]

        rh_distance = euclidean_distance(rh_right, rh_left)
        rv_distance = euclidean_distance(rv_top, rv_bottom)

        ratio_ = rh_distance / rv_distance
        return ratio_

    def detect_liveness_by_eye_mouth_movement(self, landmarks_coords_):
        """
        Detect the liveness of the user by detecting the eye and mouth movement using the eye aspect ratio and the mouth
        aspect ratio.
        :param landmarks_coords_: landmark pixels coordinates extracted from the face
        :return: boolean value of the liveness of the user
        """
        is_mouth_open = is_eye_closed = False

        # calculate the mouth aspect ratio
        mar = self.mouth_ratio(landmarks_coords_, self.UPPER_LIPS, self.LOWER_LIPS)

        # check if the mouth is open
        if mar < self.mouth_threshold:
            self.CMF_COUNTER += 1
        else:
            if self.CMF_COUNTER >= self.OPENED_MOUTH_FRAME:
                self.TOTAL_MOUTH_OPEN += 1
                self.CMF_COUNTER = 0
                is_mouth_open = True

        # calculate the eye aspect ratio
        ear = self.blink_ratio(landmarks_coords_, self.RIGHT_EYE, self.LEFT_EYE)

        if ear > self.eye_threshold:
            self.CEF_COUNTER += 1
        else:
            if self.CEF_COUNTER >= self.CLOSED_EYES_FRAME:
                self.TOTAL_BLINKS += 1
                self.CEF_COUNTER = 0
                is_eye_closed = True

        if is_eye_closed or is_mouth_open:
            return True
        else:
            return False


class FaceLiveness_Colorspace_Print_Replay_Attack:
    """
    This class is used to detect the liveness of the user by using the color spaces and the histograms of the images.
    """
    def __init__(self, path_):
        """
        Initialize the classifier and load the model.
        :param path_: path of the model classifier to be loaded
        """
        # load the classifier
        self.clf = joblib.load(path_)

    # convert the images from RGB to YCrCb and CIE L*u*v* color spaces
    @staticmethod
    def convert_color_spaces(images):
        """
        Convert the images from RGB to YCrCb and CIE L*u*v* color spaces
        :param images: list of images to be converted to the color spaces
        :return: list of images converted to the color spaces YCrCb and CIE L*u*v*
        """
        images_ycrcb = []
        images_luv = []
        for image in images:
            image_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            image_luv = cv2.cvtColor(image, cv2.COLOR_BGR2Luv)
            images_ycrcb.append(image_ycrcb)
            images_luv.append(image_luv)

        return images_ycrcb, images_luv

    # extract the histograms from the images
    @staticmethod
    def extract_histograms(images):
        """
        Extract the histograms from the images in the YCrCb and CIE L*u*v* color spaces and concatenate them.
        :param images: list of YCrCb and CIE L*u*v* images
        :return: list of concatenated histograms
        """
        histograms = []
        for image in images:
            # Y CrCb
            y, cr, cb = cv2.split(image)
            hist_y = cv2.calcHist([y], [0], None, [256], [0, 256])
            hist_cr = cv2.calcHist([cr], [0], None, [256], [0, 256])
            hist_cb = cv2.calcHist([cb], [0], None, [256], [0, 256])
            # CIE L*u*v*
            l, u, v = cv2.split(image)
            hist_l = cv2.calcHist([l], [0], None, [256], [0, 256])
            hist_u = cv2.calcHist([u], [0], None, [256], [0, 256])
            hist_v = cv2.calcHist([v], [0], None, [256], [0, 256])

            # Next, the six histograms are concatenated into a Feature Vector F V = (Y, Cr, Cb, L, u, v)
            # of size 1536 (six normalized histograms in the range of 0–255) that serve as input for
            # the Extra Trees Classifier - ETC
            hist = np.concatenate((hist_y, hist_cr, hist_cb, hist_l, hist_u, hist_v), axis=0)
            histograms.append(hist)

        return histograms

    # normalize the histograms
    @staticmethod
    def normalize_histograms(histograms):
        """
        Normalize the histograms to the range of 0–1. This is done to avoid the bias of the classifier towards the
        larger values of the histogram.
        :param histograms: list of histograms to be normalized
        :return: list of normalized histograms
        """
        normalized_histograms = []
        for histogram in histograms:
            normalized_histogram = cv2.normalize(histogram, histogram, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                                 dtype=cv2.CV_32F)
            normalized_histograms.append(normalized_histogram)
        return normalized_histograms

    # extract the features from the images
    def extract_features(self, faces_):
        """
        Extract the features from the images in the YCrCb and CIE L*u*v* color spaces, calculate the histograms and
        normalize them.
        :param faces_: list of images to be extracted the features
        :return: list of features extracted from the images as normalized histograms
        """
        images_yuv, images_yuv2 = self.convert_color_spaces(faces_)
        histograms = self.extract_histograms(images_yuv)
        normalized_histograms = self.normalize_histograms(histograms)
        return normalized_histograms

    # predict the liveness of the face image
    def predict_liveness(self, face_):
        """
        Predict the liveness of the given face image.
        :param face_: face image to be predicted the liveness
        :return: the probability of the prediction and the class label of the prediction
        """
        feature_vector = self.extract_features(face_)
        feature_vector = feature_vector[0].reshape(-1, feature_vector[0].shape[0])
        predictions = self.clf.predict_proba(feature_vector)

        # get max probability
        max_prob = np.max(predictions)
        # get the index of the max probability
        max_prob_index = np.argmax(predictions)
        # get the class label of the max probability
        max_prob_class = self.clf.classes_[max_prob_index]

        return max_prob, max_prob_class
