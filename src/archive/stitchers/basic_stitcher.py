"""
------ NOT USED IN THE MAIN SCRIPT ------
Stitcher for two images.
"""

import logging

import cv2
import numpy as np


class Stitcher:
    def __init__(self):
        pass

    def detectAndDescribe(self, image):

        # detect and extract features from the image
        descriptor = cv2.SIFT_create()
        kps, features = descriptor.detectAndCompute(image, None)

        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
        # compute the raw matches and initialize the list of actual
        # matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        for m, n in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if m.distance < n.distance * ratio:
                matches.append(m)

        # computing a homography requires at least 4 matches
        if len(matches) > 4:

            ptsA = np.float32([kpsA[match.queryIdx].pt for match in matches])
            ptsB = np.float32([kpsB[match.trainIdx].pt for match in matches])

            (H, status) = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, reprojThresh)

            return (matches, H, status)

        return None

    def stitch(self, images, ratio=0.75, reprojThresh=4.0, showMatches=False):

        logging.info("Detecting and describing keypoints...")
        imageA, imageB = images
        kpsA, featuresA = self.detectAndDescribe(imageA)
        kpsB, featuresB = self.detectAndDescribe(imageB)

        logging.info("Matching features...")
        # match features between the two images
        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

        # if the match is None, then there aren't enough matched
        # keypoints to create a panorama
        if M is None:
            return None

        logging.info("Applying homography...")
        # otherwise, apply a perspective warp to stitch the images together
        matches, H, _ = M
        result = cv2.warpPerspective(
            imageB, H, (imageB.shape[1] + imageA.shape[1], imageB.shape[0])
        )
        result[0 : imageA.shape[0], 0 : imageA.shape[1]] = imageA

        if showMatches:
            logging.info("Building matches image...")
            img_matches = np.empty(
                (max(imageA.shape[0], imageB.shape[0]), imageA.shape[1] + imageB.shape[1], 3),
                dtype=np.uint8,
            )
            cv2.drawMatches(
                imageA,
                kpsA,
                imageB,
                kpsB,
                matches,
                img_matches,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )

            return (result, img_matches)

        # return the stitched image
        return result
