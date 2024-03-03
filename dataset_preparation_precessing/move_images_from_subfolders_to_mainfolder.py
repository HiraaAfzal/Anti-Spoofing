import os
import random


def move_images_to_mainfolder(from_folder, to_folder):
    if not os.path.exists(to_folder):
        os.makedirs(to_folder)

    # get all sub_folders in from_folder
    sub_folders = [from_folder + "\\" + sub_folder for sub_folder in os.listdir(from_folder)]

    counter = 1
    # loop over the sub_folders
    for sub_folder in sub_folders:
        # get all images in sub_folder
        images = [sub_folder + "\\" + image for image in os.listdir(sub_folder)]

        # loop over the images
        for image in images:
            # move the image to to_folder
            os.rename(image, to_folder + "\\" + image.split("\\")[-1].split(".")[0] + "_" + str(counter) + ".jpg")
            counter += 1


# write a method to move number of images from a folder to another folder randomly
def move_images_to_mainfolder_randomly(from_folder, to_folder, number_of_images):
    if not os.path.exists(to_folder):
        os.makedirs(to_folder)

    counter = 1
    # get all images in sub_folder
    images = [from_folder + "\\" + image for image in os.listdir(from_folder)]

    # shuffle the images
    random.shuffle(images)

    # loop over the images
    for image in images:
        # move the image to to_folder
        os.rename(image, to_folder + "\\" + image.split("\\")[-1].split(".")[0] + "_" + str(counter) + ".jpg")
        counter += 1
        if counter > number_of_images:
            return


if __name__ == '__main__':
    from_folder_ = r"C:\Users\adnan\PycharmProjects\SIV-Project-AntiSpoofing\dataset_preparation_precessing\custom_dataset\processed\Real"
    to_folder_ = r"C:\Users\adnan\PycharmProjects\SIV-Project-AntiSpoofing\dataset_preparation_precessing\custom_dataset\processed\Real_randomly"

    print("Moving images...")
    move_images_to_mainfolder_randomly(from_folder_, to_folder_, 2000)
    print("Done")
