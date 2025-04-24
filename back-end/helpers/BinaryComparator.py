# import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageChops



class BinaryComparator:
    def __init__(self):
        pass
   
    def __openImage(self, path):
        # TODO : there are other libraries to open images. What is the best one?
        try:
            img = Image.open(path)
            return img
        except:
            raise Exception("Could not find image path: ", path)
    
    def _compareDimensions(self, img1, img2):
        # compare the height and width of the images
        try:
            height_a, width_a = img1.shape[:2]
            height_b, width_b = img2.shape[:2]
        except:
            height_a, width_a = img1.size
            height_b, width_b = img2.size
        if height_a != height_b or width_a != width_b:
            return False
        return True
    
    def __img2gray(self, img):
        # convert an image to grayscale
        # multiply rgb channels by the following weights
        grayImg = np.dot(img[...,0:3], [0.299, 0.587, 0.114])
        return grayImg
    
    def plotImage(self, img, grayscale=False):
        if grayscale:
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(img)
        
        plt.show()

    def plotComparisons(self, img1, img2, diff):
        plt.subplot(1,3,1)
        plt.imshow(img1)
        plt.title("img1")

        plt.subplot(1,3,2)
        plt.imshow(img2)
        plt.title("img2")

        plt.subplot(1,3,3)
        plt.imshow(diff)
        plt.title("difference")

        plt.show()

    def __checkColorSchemes(self, img1, img2):
        # If the images have the same mode (i.e. grayscale or rgb image), return true
        return img1.mode == img2.mode
    
    def compareImages(self, path1, path2):
        if path1 is None or path2 is None:
            raise Exception("You need to specify image paths")

        # Open images
        img1 = self.__openImage(path1)
        img2 = self.__openImage(path2)

        # Compare height and width
        sameDimensions = self._compareDimensions(img1, img2)
        if not sameDimensions:
            return False
        
        # Compare color schemes
        sameColor = self.__checkColorSchemes(img1, img2)
        if not sameColor:
            print("Images are not the same color scheme. Converting to grayscale")

            # Convert images to grayscale
            img1 = img1.convert('L')
            img2 = img2.convert('L')

        try:
            # Optionally convert to rgb
            # img1 = img1.convert('RGB')
            # img2 = img2.convert('RGB')
            difference = ImageChops.difference(img1, img2)
            if difference.getbbox() is None:
                return True
            # self.plotImage(difference)
            #self.plotComparisons(img1, img2, difference)
            return False
            
        except Exception as e:
            print(f'Error comparing images: {e}')
            return False
