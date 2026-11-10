import sys
import argparse
import cv2
import numpy as np
from PIL import Image

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("-i", "--interval", type=float, default=1)
    ap.add_argument("--corner-size", type=int, default=50)
    ap.add_argument("--white-value", type=float, default=250.0)
    return ap.parse_args()


def is_slide(
    frame: np.ndarray, 
    square_size: int = 50,
    min_mean_brightness: float = 254.0,
) -> bool:
    """
    Current heuristic for this is just checking if the top left 
    corner is a white square in essence.
    But its a video, so some compression artifacts and what not.
    So it seems easier to do this with greyscale.
    """
    patch = frame[0:square_size, 0:square_size]
    gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    mean_val = np.mean(gray_patch)
    std_val = np.std(gray_patch)
    return (mean_val >= min_mean_brightness) and std_val < 10.0

def screenshot(frame: np.ndarray) -> Image:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)
    return pil_img

def main():
    """
    sample the video at elapsed interval
    check if its a slide
    screenshot the slide and add to slides
    concatenate into PDF
    output to PDF
    """
    args = parse_args()
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Could not open {args.video}")
    
    sample_interval = args.interval
    slide_images = []
    frame_idx = -1
    no_recent_white = True
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % sample_interval == 0:
            if is_slide(frame, args.corner_size, args.white_value):
                if not no_recent_white:
                    continue
                no_recent_white = False
                slide_images.append(screenshot(frame))
            else:
                no_recent_white = True
        frame_idx += 1
    
    cap.release()
    if slide_images:
        first_page = slide_images[0]
        rest = slide_images[1:]
        first_page.save(
            "output.pdf",
            save_all=True,
            append_images=rest,
            resolution=100.0
        )
        print(f"Extracted {len(slide_images)} slides")
    else:
        print("No frames were detected") 


if __name__ == "__main__":
    main()