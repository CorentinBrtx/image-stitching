"""
------ NOT USED IN THE MAIN SCRIPT ------
Stitching with OpenCV stitcher.
"""

import argparse
import logging

import cv2
from imutils import paths

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i", "--images", type=str, required=True, help="path to input directory of images to stitch"
)
parser.add_argument("-o", "--output", type=str, required=True, help="path to the output image")
parser.add_argument("-v", "--verbose", action="store_true", help="increase output verbosity")
args = vars(parser.parse_args())


if args["verbose"]:
    logging.basicConfig(level=logging.INFO)

logging.info("Loading images...")
images = []
imagePaths = paths.list_images(args["images"])
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    images.append(image)

logging.info("Stitching images...")
stitcher = cv2.Stitcher_create()
(status, stitched) = stitcher.stitch(images)


if status == 0:
    logging.info("Stiching successful, saving result...")
    cv2.imwrite(args["output"], stitched)
else:
    logging.info(f"Image stitching failed ({status})")
