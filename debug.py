import sys

import cv2

sys.path.append("core")
from utils import flow_viz
from utils import frame_utils


def main():
    in_file = "output/frame_0016.flo"
    out_file = "frame_0016.png"

    flo = frame_utils.readFlow(in_file)
    flo = flow_viz.flow_to_image(flo)
    cv2.imwrite(out_file, flo)


if __name__ == "__main__":
    main()
