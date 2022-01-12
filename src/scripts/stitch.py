import argparse

import cv2
import logging

from ..stitchers.stitcher import Stitcher

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--first", required=True, help="path to the first image")
parser.add_argument("-s", "--second", required=True, help="path to the second image")
parser.add_argument("-o", "--output", type=str, required=True, help="path to the output image")
args = vars(parser.parse_args())

imageA = cv2.imread(args["first"])
imageB = cv2.imread(args["second"])
# imageA = imutils.resize(imageA, width=400)
# imageB = imutils.resize(imageB, width=400)

logging.basicConfig(level=logging.INFO)

stitcher = Stitcher()
result, matches_img = stitcher.stitch([imageA, imageB], showMatches=True)

cv2.imshow("Image A", imageA)
cv2.imshow("Image B", imageB)
cv2.imshow("Keypoint Matches", matches_img)
cv2.imshow("Result", result)
cv2.imwrite(args["output"], result)
cv2.waitKey(0)
