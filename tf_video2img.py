"""
输入：video base 文件夹 -- 多个video目录
输出: 图片文件夹

want : 每0.5秒抽一帧
"""
import cv2

def may_mkdir(dir_name):
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

video_dir_base = "./3.27"
img_output_base = "./3.27_img"
for video_dir_name in os.listdir(video_dir_base):
    print("*"*80)
    print(video_dir_name)
    video_path = os.path.join(video_dir_base, video_dir_name)
    output_path = os.path.join(img_output_base, video_dir_name)
    may_mkdir(output_path)

    cap = cv2.VideoCapture(video_path)

    # 总帧数
    frame_counter = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 帧率
    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))

    # 每0.5秒抽一帧
    want = 0.5
    time_interval = want * fps

    rval=cap.isOpened()
    c=0
    while rval:
    	c = c + 1
    	ret, frame = cap.read()
    	if c%time_interval == 0 and ret == True:
            print(c)
            output_name = os.path.join(output_path, str(c)+'.jpg')
    		cv2.imwrite(output_name, frame)
    cap.release()