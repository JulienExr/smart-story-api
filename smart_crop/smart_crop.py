import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForImageSegmentation
from PIL import Image, ImageDraw
import torch.nn.functional as F
from torchvision import transforms
from facenet_pytorch import MTCNN
import warnings
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from simul_api import api_simul, file_to_image

from curator.curator import get_top_images

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.serialization")

def setup_saliency_model(device, dtype):
    """Charge le modèle IA BiRefNet pour détourer le sujet."""


    print(f"Chargement du modèle de saillance...")
    
    model = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet", 
        trust_remote_code=True, 
        torch_dtype=dtype,
        weights_only=True
    ).to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return transform, model, device, dtype

def get_saliency_map(image, model, transform, device, dtype):
    
    original_size = image.size

    input_tensor = transform(image.convert('RGB')).unsqueeze(0).to(device, dtype=dtype)

    with torch.no_grad():
        outputs = model(input_tensor)
        
        if isinstance(outputs, (list, tuple)):
            result_tensor = outputs[-1]
        else:
            result_tensor = outputs

        result_tensor = torch.sigmoid(result_tensor)
        result_tensor = F.interpolate(result_tensor, size=(original_size[1], original_size[0]), mode='bilinear', align_corners=False)
        
        ma = torch.max(result_tensor)
        mi = torch.min(result_tensor)
        result_tensor = (result_tensor - mi) / (ma - mi + 1e-8)
        
        mask_np  = (result_tensor.squeeze().cpu().float().numpy() * 255).astype(np.uint8)
    return Image.fromarray(mask_np)


def get_group_faces(image, device):
    
    face_detector = MTCNN(keep_all=True, device=device)
    boxes, probs = face_detector.detect(image)

    print(f"Nombre de visages détectés : {len(probs) if probs is not None else 0}")
    print(f"Probabilités de détection : {probs}")
    if boxes is None:
        return None

    new_boxes = []
    for box, prob in zip(boxes, probs):
        if prob < 0.90:
            print(f"Visage rejeté avec probabilité : {prob}")
        else:
            new_boxes.append(box)
    boxes = new_boxes
    print(f"Nombre de visages retenus après filtrage : {len(boxes)}")
    faces = []

    for box in boxes:
        x_min, y_min, x_max, y_max = box
        w = x_max - x_min
        h = y_max - y_min
        area = w * h
        faces.append({"x": int(x_min), "y": int(y_min), "w": int(w), "h": int(h), "area": int(area)})
    
    max_area = max(f["area"] for f in faces)
    significant_faces = [f for f in faces if f["area"] > 0.25 * max_area]
    print(f"Nombre de visages significatifs : {len(significant_faces)}")

    return significant_faces


def get_faces_map(image, device):
    faces = get_group_faces(image, device)
    if not faces:
        return None, None
    
    img_np = np.array(image)
    H, W, _ = img_np.shape
    mask = np.zeros((H, W), dtype=np.uint8)
    for f in faces:
        x, y, w, h = f["x"], f["y"], f["w"], f["h"]
        mask[y:y+h, x:x+w] = 255
    return Image.fromarray(mask), faces

def draw_faces_on_image(image, faces):
    boxed = image.convert("RGB").copy()
    draw = ImageDraw.Draw(boxed)
    for f in faces:
        x, y, w, h = f["x"], f["y"], f["w"], f["h"]
        draw.rectangle([x, y, x + w, y + h], outline="lime", width=3)
    return boxed

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

def plot_faces_map(image, faces_mask, faces=None):
    """plot the original image and his faces map side by side"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    if faces:
        ax1.imshow(draw_faces_on_image(image, faces))
        ax1.set_title("Original Image + Face Boxes")
    else:
        ax1.imshow(image)
        ax1.set_title("Original Image")

    ax1.axis('off')

    if faces_mask is None:
        empty_mask = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)
        ax2.imshow(empty_mask, cmap='gray')
        ax2.set_title("Faces Map (no face detected)")
    else:
        ax2.imshow(faces_mask, cmap='gray')
        ax2.set_title("Faces Map")

    ax2.axis('off')
    plt.show()


if __name__ == "__main__":  
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    images = api_simul("smart_crop/data")
    images = get_top_images(images, 10) 
    
    transform, model, device, dtype = setup_saliency_model(device, dtype)

    for img in images:
        saliency_mask = get_saliency_map(img, model, transform, device, dtype)
        faces_mask, faces = get_faces_map(img, device)

        plot_saliency_map(img, saliency_mask)
        plot_faces_map(img, faces_mask, faces)