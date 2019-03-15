import shutil
import os
import cv2


def copy_file(src1):
    dst = "/data/zl/object_detection/images/"
    src = "/data/zl/南瑞项目素材收集-新版/01.大型机械/00.图片/02.湖北现场"
    files = os.listdir(src1)
    for file in files:
        print(file)
        shutil.copy(os.path.join(src, file), dst)

def video_to_images(video_dir, image_dir, prefix="images", start=0, times=1):
    for file in os.listdir(video_dir):
        cap = cv2.VideoCapture(os.path.join(video_dir, file))
        print(file,cap.get(cv2.CAP_PROP_FPS))
        frame_num = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1
            if frame_num % int(cap.get(cv2.CAP_PROP_FPS) / times) == 0:
                cv2.imwrite(os.path.join(image_dir, "{}_{}.jpg".format(prefix, str(start).zfill(5))), frame)
                start += 1
        cap.release()


if __name__ == '__main__':
    video_to_images("/data/zl/南瑞项目素材收集-新版/01.大型机械/02.视频/02.湖北现场", "/data/zl/南瑞项目素材收集-新版/01.大型机械/images/",
                    "dxjx_sd_20190315",start=300)
