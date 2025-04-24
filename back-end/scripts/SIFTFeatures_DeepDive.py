# this file will be the same as the SIFTFeatures.py, but will compare every photo to every other photo in the near duplicates dataset
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import time

root = os.getcwd()
folderRoute = "nextjs-flask\\public\\duplicates\\near-duplicates\\"
basePath = os.path.join(root, folderRoute)

def crawlFolder(folderPath):
    file_info_list = []
    
    # Convert folder path to Path object for easier manipulation
    base_path = Path(folderPath)
    
    # Traverse through all files and directories in folder_path
    for root, dirs, files in os.walk(base_path):
        for file in files:
            file_path = Path(root) / file
            file_info = {
                "file_name": file,
                "file_path": str(file_path),
                "file_size": file_path.stat().st_size  # File size in bytes
            }
            file_info_list.append(file_info)
    
    return file_info_list

file_info_list = crawlFolder(basePath)
# print(len(file_info_list))

def SIFT_Single(image):
    sift = cv2.SIFT_create()
    im_path = image['file_path']
    im = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
    keypoints, descriptors = sift.detectAndCompute(im, None)
    output_image = cv2.drawKeypoints(
        im, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    cv2.imshow("SIFT Keypoints", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

SIFT_Single(file_info_list[0])

def SIFTCompareAll(fileInfoList):
    numMatches = [[0 for _ in range(len(fileInfoList))] for _ in range(len(fileInfoList))]
    ratios = [[0.0 for _ in range(len(fileInfoList))] for _ in range(len(fileInfoList))]
    times = [[0 for _ in range(len(fileInfoList))] for _ in range(len(fileInfoList))]
    sift = cv2.SIFT_create()
    keypoints = {}
    descriptors = {}
    
    # TODO memoize this so that you are not repeating work
    for i in range(len(fileInfoList)):
        # get first image
        if i in descriptors:
            keypoints1 = keypoints[i]
            descriptors1 = descriptors[i]
        else:
            im1_info = fileInfoList[i]
            im1Path = im1_info['file_path']

            im1 = cv2.imread(im1Path, cv2.IMREAD_UNCHANGED)

            keypoints1, descriptors1 = sift.detectAndCompute(im1, None)
            keypoints[i] = keypoints1
            descriptors[i] = descriptors1
        for j in range(i, len(fileInfoList)):
            # get second image
            if j in descriptors:
                keypoints2 = keypoints[j]
                descriptors2 = descriptors[j]
            else:
                im2_info = fileInfoList[j]
                im2Path = im2_info['file_path']

                im2 = cv2.imread(im2Path, cv2.IMREAD_UNCHANGED)

                keypoints2, descriptors2 = sift.detectAndCompute(im2, None)
                keypoints[j] = keypoints2
                descriptors[j] = descriptors2

            # brute force match
            start = time.time()
            bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
            print(i, j)
            try:
                matches = bf.match(descriptors1, descriptors2)
                ratio = len(matches) / len(descriptors1)
            except cv2.error as e:
                print("Error during matching:", e)
                matches = []  # Handle as needed
                ratio = 0.0
            # matches = bf.match(descriptors1, descriptors2)
            # matches = sorted(matches, key=lambda x:x.distance)
            end = time.time()

            # update numMatches
            numMatches[i][j] = len(matches)
            ratios[i][j] = ratio
            times[i][j] = end - start
    return numMatches, ratios, times

from itertools import islice


# numMatches, ratios, times = SIFTCompareAll(file_info_list[:10])

def plot_heatmap(numMatches, title="Number of Matches"):
    plt.figure(figsize=(10,8))
    sns.heatmap(numMatches, annot=True, fmt="d", cmap="YlGnBu", cbar=True)

    plt.title(title)
    plt.xlabel("File Index")
    plt.ylabel("File Index")
    plt.show()

def plot_heatmap_float(numMatches, title="Ratios"):
    plt.figure(figsize=(10,8))
    sns.heatmap(numMatches, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True)

    plt.title(title)
    plt.xlabel("File Index")
    plt.ylabel("File Index")
    plt.show()

def plot_heatmap_times(times, title="Times to Compare"):
    plt.figure(figsize=(10,8))
    sns.heatmap(times, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True)

    plt.title(title)
    plt.xlabel("File Index")
    plt.ylabel("File Index")
    plt.show()

# plot_heatmap(numMatches)
# plot_heatmap_times(times)