# import the necessary packages
import argparse

import cv2
from imutils import paths

# construct the argument parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "-i", "--images", type=str, required=True, help="path to input directory of images to stitch"
)
parser.add_argument("-o", "--output", type=str, required=True, help="path to the output image")
args = vars(parser.parse_args())

print("[INFO] loading images...")
images = []
imagePaths = paths.list_images(args["images"])
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    images.append(image)

print("[INFO] stitching images...")
stitcher = cv2.Stitcher_create()
(status, stitched) = stitcher.stitch(images)


if status == 0:
    print("[INFO] stiching successful, saving and showing result...")
    cv2.imwrite(args["output"], stitched)
    cv2.imshow("Stitched", stitched)
    cv2.waitKey(0)
else:
    print(f"[INFO] image stitching failed ({status})")
