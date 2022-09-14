import moviepy.editor as mp
import os
import glob

def gif2mp4(gif_dir, mp4_dir):
    gifs = glob.glob(os.path.join(gif_dir, "*.gif"))
    for gif in gifs:
        clip = mp.VideoFileClip(gif)
        target_path = os.path.join(mp4_dir, os.path.basename(gif).replace(".gif", ".mp4"))
        clip.write_videofile(target_path)


if __name__ == "__main__":
    gif2mp4("path/to/gifs", "path/to/mp4")