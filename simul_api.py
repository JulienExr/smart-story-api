from PIL import Image
import os


def directory_to_list(folder_path):
    image_files = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_files.append(filename)
    return image_files

def file_to_image(image_path):
    return Image.open(image_path)

def api_simul(directory_path):
    image_files = directory_to_list(directory_path)
    images = [file_to_image(os.path.join(directory_path, img)) for img in image_files]
    return images