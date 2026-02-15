import cv2
import torch
import clip
import imagehash
import torch.nn.functional as F
from PIL import Image
from AestheticPredicteur import AestheticPredictor
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os
import sys
from tqdm import tqdm
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from simul_api import api_simul


MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))

SEUIL_REJET_BRISQUE = 50.0
SEUIL_MINIMUM_LAION = 5.0
TARGET_WIDTH = 720

def PIL_to_cv2(image):
    image = image.convert("RGB")
    open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return open_cv_image

def get_brisque_score(image):
    model_path = os.path.join(MODEL_DIR, "brisque_model_live.yml")
    range_path = os.path.join(MODEL_DIR, "brisque_range_live.yml")

    if not os.path.exists(model_path) or not os.path.exists(range_path):
        print(f"Error: BRISQUE model files not found at {model_path} and {range_path}")
        return None
    
    img = PIL_to_cv2(image)

    if img is None:
        print(f"Error: Could not read image at {image}")
        return None

    h, w = img.shape[:2]
    ratio = TARGET_WIDTH / w
    new_h = int(h * ratio)
    img = cv2.resize(img, (TARGET_WIDTH, new_h), interpolation=cv2.INTER_AREA)
    brisque_alg = cv2.quality.QualityBRISQUE_create(model_path, range_path)
    score = brisque_alg.compute(img)[0]
    
    return score

def setup_aesthetic_predictor():
    CLIP_MODEL_NAME = "ViT-L/14"
    AESTHETIC_WEIGHTS_PATH = os.path.join(MODEL_DIR, "sac+logos+ava1-l14-linearMSE.pth")
    if not os.path.exists(AESTHETIC_WEIGHTS_PATH):
        raise FileNotFoundError(f"Aesthetic weights not found: {AESTHETIC_WEIGHTS_PATH}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading CLIP model: {CLIP_MODEL_NAME}...")
    model_clip, preprocess = clip.load(CLIP_MODEL_NAME, device=device)
    print("CLIP model loaded successfully.")

    print(f"Loading aesthetic predictor weights from: {AESTHETIC_WEIGHTS_PATH}...")
    pt_state = torch.load(AESTHETIC_WEIGHTS_PATH, map_location=device)
    predictor = AestheticPredictor(768)
    predictor.load_state_dict(pt_state)
    predictor.to(device)
    predictor.eval()
    print("Aesthetic predictor loaded and ready.")
    return model_clip, preprocess, predictor, device

def predict_aesthetic_score(image, model_clip, preprocess, predictor, device):

    image = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model_clip.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.float()
        
        score = predictor(image_features)

    return score.item()

def get_aesthetic_score(image, model_clip, preprocess, predictor, device):
    score = predict_aesthetic_score(image, model_clip, preprocess, predictor, device)
    return score

def get_combined_score(brisque_score, aesthetic_score):
    
    aesthetic_score *= 10
    brisque_score = 100 - brisque_score

    combined_score = 0.5 * aesthetic_score + 0.5 * brisque_score
    return combined_score

def check_similarity_clip(image1, image2, model_clip, preprocess, device):
    image1 = preprocess(image1).unsqueeze(0).to(device)
    image2 = preprocess(image2).unsqueeze(0).to(device)

    with torch.no_grad():
        features1 = model_clip.encode_image(image1)
        features2 = model_clip.encode_image(image2)

        features1 = features1 / features1.norm(dim=-1, keepdim=True)
        features2 = features2 / features2.norm(dim=-1, keepdim=True)

        sim = F.cosine_similarity(features1, features2).item()
        print(f"CLIP Similarity: {sim * 100 :.4f}")
    return sim

def check_similarity_hash(image1, image2):
    hash1 = imagehash.phash(image1)
    hash2 = imagehash.phash(image2)

    distance = hash1 - hash2
    print(f"Hash Distance: {distance}")
    return distance

def pretty_filename(filename):
    if isinstance(filename, (list, tuple)):
        return [os.path.basename(item) for item in filename]
    return os.path.basename(filename)

def remove_duplicates(image_list):
    hashes = {}
    for image in image_list:
        filename = image.filename
        img_hash = imagehash.phash(image)

        if img_hash in hashes:
            print(f"Duplicate found: {pretty_filename(filename)} is a duplicate of {pretty_filename(hashes[img_hash])}")
            image_list.remove(image)
        else:
            hashes[img_hash] = filename

def remove_similar_images(image_list, model_clip, preprocess, device, threshold=0.9, min_imgs=5):
    features = {}
    for image in image_list:
        filename = image.filename
        image = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model_clip.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            features[filename] = image_features

    for filename1, feat1 in features.items():
        for filename2, feat2 in features.items():
            if len(image_list) <= min_imgs:
                return
            
            if filename1 >= filename2:
                continue
            sim = F.cosine_similarity(feat1, feat2).item()
            if sim > threshold:
                print(f"Similar images: {pretty_filename(filename1)} and {pretty_filename(filename2)} (Similarity: {sim * 100:.4f})")
                score1 = get_brisque_score(Image.open(filename1))
                score2 = get_brisque_score(Image.open(filename2))
                if score1 is not None and score2 is not None:
                    if score1 < score2:
                        print(f"Removing {pretty_filename(filename1)} (BRISQUE: {score1:.2f} vs {score2:.2f})")
                        image_list.remove(Image.open(filename1))
                    else:
                        print(f"Removing {pretty_filename(filename2)} (BRISQUE: {score2:.2f} vs {score1:.2f})")
                        image_list.remove(Image.open(filename2))

def clasify_images(image_list, model_clip, preprocess, predictor, device, min_imgs=5):

    remove_duplicates(image_list)
    print("Duplicates removed.")

    remove_similar_images(image_list, model_clip, preprocess, device, threshold=0.90, min_imgs=min_imgs)
    print("Similar images removed.")

    brisque_scores = {}
    print("Calculating BRISQUE scores...")
    
    total_files = len(image_list)
    for image in tqdm(image_list, desc="Analyse BRISQUE", unit="img"):
        filename = image.filename
        brisque_score = get_brisque_score(image)
        brisque_scores[filename] = brisque_score


    filenames_cleaned = [f for f in brisque_scores.keys() if brisque_scores[f] is not None and brisque_scores[f] < SEUIL_REJET_BRISQUE]
    aesthetic_scores = {}

    print(f"\nCalculating aesthetic scores...")
    for filename in tqdm(filenames_cleaned, desc="Analyse Aesthetic", unit="img"):
        image = Image.open(filename)
        aesthetic_score = get_aesthetic_score(image, model_clip, preprocess, predictor, device)
        aesthetic_scores[filename] = aesthetic_score

    final_filenames = [f for f in aesthetic_scores.keys() if aesthetic_scores[f] > SEUIL_MINIMUM_LAION]
    final_scores = {}
    print(f"\nCalculating combined scores...")
    for filename in final_filenames:
        combined_score = get_combined_score(brisque_scores[filename], aesthetic_scores[filename])
        final_scores[filename] = combined_score
    print("Classification completed.")
    return brisque_scores, aesthetic_scores, final_scores

def get_top_images(scores, top_n=5):
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_images = sorted_scores[:top_n]
    return top_images

def plot_scores(scores, folder_path):

    filenames = list(scores.keys())
    score_values = list(scores.values())
    
    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(range(len(filenames)), score_values, color='blue', alpha=0.6)
    
    for i, (filename, score) in enumerate(zip(filenames, score_values)):
        image_path = os.path.join(folder_path, filename)
        try:
            img = Image.open(image_path)
            img.thumbnail((60, 60))
            
            imagebox = OffsetImage(img, zoom=1)
            ab = AnnotationBbox(imagebox, (i, score), 
                                xybox=(0, 20),
                                xycoords='data',
                                boxcoords="offset points",
                                frameon=True,
                                pad=0.3)
            ax.add_artist(ab)
        except Exception as e:
            print(f"Erreur lors du chargement de {filename}: {e}")
    
    ax.set_xticks(range(len(filenames)))
    ax.set_xticklabels(pretty_filename(filenames), rotation=45, ha='right')
    ax.set_xlabel('Image Filename')
    ax.set_ylabel('Score')
    ax.set_title('Scores for Images in Folder')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Setting up aesthetic predictor...")
    model_clip, preprocess, predictor, device = setup_aesthetic_predictor()

    folder_path = DATA_DIR

    image_list = api_simul(folder_path)
    print(f"Total images loaded: {len(image_list)}")

    brisque_scores, aesthetic_scores, final_scores = clasify_images(image_list, model_clip, preprocess, predictor, device)
    brisque_scores = dict(sorted(brisque_scores.items(), key=lambda item: item[1], reverse=True))
    aesthetic_scores = dict(sorted(aesthetic_scores.items(), key=lambda item: item[1], reverse=True))
    final_scores = dict(sorted(final_scores.items(), key=lambda item: item[1], reverse=True))
    
    for filename in final_scores.keys():
        print(
            f"Selected: {pretty_filename(filename)} - Aesthetic Score: {aesthetic_scores[filename]:.2f}, BRISQUE Score: {brisque_scores[filename]:.2f}"
        )
    plot_scores(brisque_scores, folder_path)
    plot_scores(aesthetic_scores, folder_path)
    plot_scores(final_scores, folder_path)

    top_images = get_top_images(final_scores, top_n=5)
    print("\nTop 5 images:")
    for filename, score in top_images:
        print(f"Selected: {pretty_filename(filename)} - Score: {score:.2f}")
