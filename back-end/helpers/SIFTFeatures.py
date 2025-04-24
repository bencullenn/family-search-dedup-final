import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def SIFT():
    root = os.getcwd()
    # print(root)
    pathRoute = "nextjs-flask\\public\\duplicates\\near-duplicates\\"
    imgName1 = "008_01.jpg"
    imgName2 = "008_02.jpg"
    imgName3 = "014_01.jpg"
    imgPath1 = os.path.join(root, pathRoute, imgName1)
    imgPath2 = os.path.join(root, pathRoute, imgName2)
    imgPath3 = os.path.join(root, pathRoute, imgName3)





    # print(imgPath)
    im1 = cv2.imread(imgPath1, cv2.IMREAD_UNCHANGED)
    im2 = cv2.imread(imgPath2, cv2.IMREAD_UNCHANGED)
    im3 = cv2.imread(imgPath3, cv2.IMREAD_UNCHANGED)

    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(im1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(im2, None)
    keypoints3, descriptors3 = sift.detectAndCompute(im3, None)

    # brute force match
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x:x.distance)

    # this line of code will draw the keypoints over the original image
    # im1 = cv2.drawKeypoints(im, keypoints, im, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    img_1_2 = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches[:50], im2, flags=2)
    plt.imshow(img_1_2)
    plt.show()
    print("008_01 : 008_02 => ", len(matches))

    
    matches_2 = bf.match(descriptors1, descriptors3)
    matches_2 = sorted(matches_2, key=lambda x:x.distance)
    img_1_3 = cv2.drawMatches(im1, keypoints1, im3, keypoints3, matches_2[:50], im3, flags=2)
    plt.imshow(img_1_3)
    plt.show()
    print("008_01 : 014_01 => ", len(matches_2))


if __name__ == '__main__':
    SIFT()