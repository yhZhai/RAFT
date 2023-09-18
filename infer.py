import argparse
import glob
import os
import sys
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
from PIL import Image

sys.path.append("core")
from raft import RAFT
from utils import frame_utils
from utils.utils import InputPadder


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None]


def save(flo, imfile1: str, output_folder: Path):
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    frame_utils.writeFlow(
        (output_folder / Path(imfile1).name).with_suffix(".flo").as_posix(), flo
    )


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location="cpu"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.module
    model.to(device)
    model.eval()

    output_dir = Path(args.output_folder)
    output_dir.mkdir(exist_ok=True)

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, "*.png")) + glob.glob(
            os.path.join(args.path, "*.jpg")
        )

        images = sorted(images)
        for imfile1, imfile2 in tqdm(
            zip(images[:-1], images[1:]), total=len(images) - 1
        ):
            image1 = load_image(imfile1).to(device)
            image2 = load_image(imfile2).to(device)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            save(flow_up, imfile1, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="restore checkpoint")
    parser.add_argument("--path", help="dataset for evaluation")
    parser.add_argument("--output_folder", type=str, default="output")
    parser.add_argument("--small", action="store_true", help="use small model")
    parser.add_argument(
        "--mixed_precision", action="store_true", help="use mixed precision"
    )
    parser.add_argument(
        "--alternate_corr",
        action="store_true",
        help="use efficent correlation implementation",
    )
    args = parser.parse_args()

    demo(args)
