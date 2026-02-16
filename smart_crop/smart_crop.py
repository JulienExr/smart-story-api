import cv2
import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from rembg import remove

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from simul_api import api_simul, file_to_image

from curator.curator import get_top_images



def get_saliency_map(image):
    saliency_mask = remove(image, only_mask=True)
    return saliency_mask

def plot_saliency_map(image, saliency_mask):
    """plot the original image and his saliency map side by side"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(image)
    ax1.set_title("Original Image")
    ax1.axis('off')
    ax2.imshow(saliency_mask, cmap='gray')
    ax2.set_title("Saliency Map")
    ax2.axis('off')
    plt.show()

if __name__ == "__main__":  

    images = api_simul("data")
    images = get_top_images(images, 5) 

    for img in images:
        saliency_mask = get_saliency_map(img)
        plt.imshow(saliency_mask, cmap='gray')
        plt.axis('off')
        plt.show()