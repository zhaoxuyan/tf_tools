import os
import cv2

def getFrame(videoPath, svPath):

    cap = cv2.VideoCapture(videoPath)

    # 总帧数
    frame_counter = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("总帧数:",frame_counter)

    # 帧率
    # fps = int(round(cap.get(cv2.CAP_PROP_FPS)))

    # 希望获得多少帧的图片
    want_frame_num = 50

    numFrame = 0
    num =0
    while numFrame < frame_counter:
        ret, frame = cap.read()
        numFrame += 1

        newPath = os.path.join(svPath, str(num) + ".jpg")
        if numFrame % (int(frame_counter / want_frame_num)) == 0:
            print(numFrame, num)
            cv2.imwrite(newPath,frame)
            num+=1

def may_mkdir(dir_name):
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)


if __name__ == '__main__':
    bath = "/disk3/xuyan.zhao/models/research/object_detection/exp/faster_rcnn_resnet101_people/add_money_video/bad"
    output_dir = "/disk3/xuyan.zhao/models/research/object_detection/exp/faster_rcnn_resnet101_people/add_money_output"
    videos = os.listdir(bath)
    for video in videos:
        video_path = os.path.join(bath,video)
        video_name = video_path.split("/")[-1]

        if video_name.endswith('dav'):
            continue

        print("*"*80)
        print(video_name)

        video_output_dir = os.path.join(output_dir, video_name[:-4])
        may_mkdir(video_output_dir)
        getFrame(video_path, video_output_dir)

