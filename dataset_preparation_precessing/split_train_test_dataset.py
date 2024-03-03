import os
import glob


# split the dataset into training and testing sets
def split_train_test_dataset(dataset_path_, train_path_, test_path_, train_size_percentage_=0.8):
    for folder in os.listdir(dataset_path_):
        images = []
        for file in glob.glob(os.path.join(dataset_path_, folder, '*.jpg')):
            images.append(file)

        train_size = int(train_size_percentage_ * len(images))

        train_images = images[:train_size]
        test_images = images[train_size:]

        # create the train and test folders if they don't exist
        if not os.path.exists(os.path.join(train_path_, folder)):
            os.makedirs(os.path.join(train_path_, folder))

        if not os.path.exists(os.path.join(test_path_, folder)):
            os.makedirs(os.path.join(test_path_, folder))

        # create class folders in the train and test folders if they don't exist
        if not os.path.exists(os.path.join(train_path_, folder)):
            os.makedirs(os.path.join(train_path_, folder))

        if not os.path.exists(os.path.join(test_path_, folder)):
            os.makedirs(os.path.join(test_path_, folder))

        for image in train_images:
            image_name = image.split(os.path.sep)[-1]
            os.rename(image, os.path.join(train_path_, folder, image_name))

        for image in test_images:
            image_name = image.split(os.path.sep)[-1]
            os.rename(image, os.path.join(test_path_, folder, image_name))

        # remove the class folder from the dataset folder
        os.rmdir(os.path.join(dataset_path_, folder))


if __name__ == '__main__':
    dataset_path = r'C:\Users\adnan\PycharmProjects\SIV-Project-AntiSpoofing\custom_dataset'
    train_path = r'C:\Users\adnan\PycharmProjects\SIV-Project-AntiSpoofing\custom_dataset\train'
    test_path = r'C:\Users\adnan\PycharmProjects\SIV-Project-AntiSpoofing\custom_dataset\test'
    print('Splitting the dataset into training and testing sets...')
    split_train_test_dataset(dataset_path, train_path, test_path, 0.8)
    print('Done!')


# # load the dataset and extract the labels and images
# def load_dataset(dataset_path_):
#     images_ = []
#     labels_ = []
#     for folder in os.listdir(dataset_path_):
#         for file in glob.glob(os.path.join(dataset_path_, folder, '*.jpg')):
#             image = cv2.imread(file)
#             images_.append(image)
#             labels_.append(folder)
#     return images_, labels_
#
#
# # convert the images from RGB to YCrCb and CIE L*u*v* color spaces
# def convert_color_spaces(images):
#     images_ycrcb = []
#     images_luv = []
#     for image in images:
#         image_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
#         image_luv = cv2.cvtColor(image, cv2.COLOR_BGR2Luv)
#         images_ycrcb.append(image_ycrcb)
#         images_luv.append(image_luv)
#
#     return images_ycrcb, images_luv
#
#
# # extract the histograms from the images
# def extract_histograms(images):
#     # 6 histograms are calculated, corresponding to each component of these two color spaces
#     # Y CrCb and CIE L*u*v*
#     histograms = []
#     for image in images:
#         # Y CrCb
#         y, cr, cb = cv2.split(image)
#         hist_y = cv2.calcHist([y], [0], None, [256], [0, 256])
#         hist_cr = cv2.calcHist([cr], [0], None, [256], [0, 256])
#         hist_cb = cv2.calcHist([cb], [0], None, [256], [0, 256])
#         # CIE L*u*v*
#         l, u, v = cv2.split(image)
#         hist_l = cv2.calcHist([l], [0], None, [256], [0, 256])
#         hist_u = cv2.calcHist([u], [0], None, [256], [0, 256])
#         hist_v = cv2.calcHist([v], [0], None, [256], [0, 256])
#
#         # Next, the six histograms are concatenated into a Feature Vector F V = (Y, Cr, Cb, L, u, v)
#         # of size 1536 (six normalized histograms in the range of 0–255) that serve as input for
#         # the Extra Trees Classifier - ETC
#         hist = np.concatenate((hist_y, hist_cr, hist_cb, hist_l, hist_u, hist_v), axis=0)
#         histograms.append(hist)
#
#     return histograms
#
#
# # extract the features from the images
# def extract_features(images):
#     images_ycrcb, images_luv = convert_color_spaces(images)
#     histograms_ycrcb = extract_histograms(images_ycrcb)
#     histograms_luv = extract_histograms(images_luv)
#     features = np.concatenate((histograms_ycrcb, histograms_luv), axis=1)
#     return features
#
#
# # train the Extra Trees Classifier
# def train_etc(features, labels):
#     etc = ExtraTreesClassifier(n_estimators=100, random_state=0)
#     etc.fit(features, labels)
#     return etc
#
#
# # test the Extra Trees Classifier
# def test_etc(etc, features, labels):
#     predictions = etc.predict(features)
#     accuracy = accuracy_score(labels, predictions)
#     return accuracy
#
#
# # main function
# if __name__ == '__main__':
#     # dataset path
#     dataset_path = 'dataset'
#     # training path
#     train_path = 'train'
#     # testing path
#     test_path = 'test'
#     # train size
#     train_size = 100
#     # split the dataset into training and testing sets
#     split_train_test_dataset(dataset_path, train_path, test_path, train_size)
#     # load the training dataset
#     images_train, labels_train = load_dataset(train_path)
#     # load the testing dataset
#     images_test, labels_test = load_dataset(test_path)
#     # extract the features from the training dataset
#     features_train = extract_features(images_train)
#     # extract the features from the testing dataset
#     features_test = extract_features(images_test)
#     # train the Extra Trees Classifier
#     etc = train_etc(features_train, labels_train)
#     # test the Extra Trees Classifier
#     accuracy = test_etc(etc, features_test, labels_test)
#     print('Accuracy: ', accuracy)
