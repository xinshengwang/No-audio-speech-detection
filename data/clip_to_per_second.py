import cv2
import os
from multiprocessing import Pool


def clip_videos(file):
    root = 'test/videos'
    save_root = 'split/test/videos/videos_per_second'
    input_path = os.path.join(root,file)
    name = file.split('.')[0]
    save_dir = os.path.join(save_root,name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cap = cv2.VideoCapture(input_path)
    cap.isOpened() 

    #get video info
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) 
    print(width, height)

    if cap.isOpened(): 
        rate = cap.get(5)  # parameter 5 : rate
        FrameNumber = int(cap.get(7)) # parameter: total frames
        duration = FrameNumber/rate 
        fps = int(rate)  
        print(rate)
        print(FrameNumber)
        print(duration)
        print(fps)

    i = 0
    while (True):
        success, frame = cap.read()
        if success:
            if (i % fps == 0): 
                num = i // fps
                videoWriter = cv2.VideoWriter(save_dir + '/' + str(num) + '.mp4', cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps, (int(width), int(height)))
                videoWriter.write(frame)  
            else:
                videoWriter.write(frame)
            i += 1
        else:
            print('end')
            break

    cap.release()  

if __name__ == "__main__":
    root = 'test/videos'
    names = os.listdir(root)
    pool = Pool(5)
    pool.map(clip_videos,names)
    pool.start()
    pool.join()