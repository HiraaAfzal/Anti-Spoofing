import os
import cv2 as cv


def convert_video_into_frames(video_path, frames_path, frame_rate=1, counter=1):
    # Create a folder to store the frames
    if not os.path.exists(frames_path):
        os.makedirs(frames_path)

    # Read the video from specified path
    cam = cv.VideoCapture(video_path)

    try:
        # creating a folder named data
        if not os.path.exists(frames_path):
            os.makedirs(frames_path)
    # if not created then raise error
    except OSError:
        print('Error: Creating directory of data')

    # frame
    currentframe = counter
    current_frame = 1
    while True:
        # reading from frame
        ret, frame = cam.read()

        if ret:
            # create frame as per frame rate
            if current_frame % frame_rate == 0:
                # if video is still left continue creating images
                name = frames_path + '/frame' + str(currentframe) + '.jpg'
                print('Creating...' + name)

                # writing the extracted images
                cv.imwrite(name, frame)

                # increasing counter so that it will
                # show how many frames are created
                currentframe += 1
            current_frame += 1
        else:
            break
        #     # if video is still left continue creating images
        #     name = frames_path + '/frame' + str(currentframe) + '.jpg'
        #     print('Creating...' + name)
        #
        #     # writing the extracted images
        #     cv.imwrite(name, frame)
        #
        #     # increasing counter so that it will
        #     # show how many frames are created
        #     currentframe += 1
        # else:
        #     break

    # Release all space and windows once done
    cam.release()
    cv.destroyAllWindows()

    return currentframe


if __name__ == '__main__':
    # name of custom dataset folder
    custom_dataset_path = 'custom_dataset'

    # path folder containing videos to be converted into frames
    videos_folder_path = r"C:\Users\adnan\Downloads\Compressed\live_video"

    # get list of videos in videos folder with full path
    videos_list = [videos_folder_path + "\\" + video for video in os.listdir(videos_folder_path)]

    # frames folder name for each video's frames
    frames_folder_name = videos_folder_path.split('\\')[-1]

    # path to frames folder
    frames_folder_path = os.path.join(os.getcwd(), custom_dataset_path, frames_folder_name)

    count = 1
    print("Converting videos into frames...")
    for video in videos_list:
        print("Processing video: ", video)
        # path to video
        video_path = os.path.join(videos_folder_path, video)

        # convert video into frames
        count = convert_video_into_frames(video_path, frames_folder_path, frame_rate=3, counter=count)

    print('Done')
