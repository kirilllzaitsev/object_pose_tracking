import os
import re
from base64 import b64encode

import cv2
from pose_tracking.utils.common import adjust_img_for_plt, create_dir


def save_video(
    images, save_path, frame_height=480, frame_width=640, fps=10, live_preview=False, live_preview_delay=1000
):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    create_dir(save_path)
    video_writer = cv2.VideoWriter(save_path, fourcc, fps, (frame_width, frame_height), isColor=True)

    for image in images:
        image = adjust_img_for_plt(image)
        image = cv2.resize(image, (frame_width, frame_height))
        video_writer.write(image)
        if live_preview:
            cv2.imshow("Video", image)
            if cv2.waitKey(live_preview_delay) & 0xFF == ord("q"):
                break

    video_writer.release()
    cv2.destroyAllWindows()
    fix_mp4_encoding(save_path)
    print(f"Video saved as {save_path}")


def save_folder_imgs_as_video(folder_path, save_path=None, **kwargs):
    images = []
    for file in sorted(os.listdir(folder_path), key=lambda x: int(re.findall(r"\d+", x)[0])):
        if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
            img = cv2.imread(os.path.join(folder_path, file))
            images.append(img)
    if save_path is None:
        save_path = os.path.join(folder_path, "video.mp4")
    save_video(images, save_path, **kwargs)


def fix_mp4_encoding(video_path):
    tmp_path = str(video_path).replace(".mp4", "_tmp.mp4")
    os.system(f"ffmpeg -i {video_path} -r 30 {tmp_path} >/dev/null 2>&1")
    os.system(f"mv {tmp_path} {video_path}")
    print("Changed video encoding to h264")


def show_video(video_path):
    from IPython.display import HTML

    video_file = open(video_path, "r+b").read()
    video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
    return HTML(f"""<video width="640" height="480" autoplay loop controls><source src="{video_url}"></video>""")
