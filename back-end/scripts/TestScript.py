import sys
import os

from BinaryComparator import BinaryComparator

path1 = ''
path2 = ''

comparator = BinaryComparator()
sameImage = comparator.compareImages(path1, path2)
if sameImage:
    print("Same image")
else:
    print("not the same image")