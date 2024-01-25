import cv2
import glob

image_path = "C:/Users/learn-admin/Desktop/UnistPy/images/*.png"

image_files = sorted(glob.glob(image_path))
images = [cv2.imread(file) for file in image_files]

video_writer = cv2.VideoWriter("../images/video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 24, images[0].shape[:2])

for image in images:
    video_writer.write(image)

video_writer.release()

print("video export")
