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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))

SEUIL_REJET_BRISQUE = 50.0
SEUIL_MINIMUM_LAION = 5.0
TARGET_WIDTH = 720

def get_brisque_score(image_path):
    model_path = os.path.join(MODEL_DIR, "brisque_model_live.yml")
    range_path = os.path.join(MODEL_DIR, "brisque_range_live.yml")

    if not os.path.exists(model_path) or not os.path.exists(range_path):
        print(f"Error: BRISQUE model files not found at {model_path} and {range_path}")
        return None
    
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Could not read image at {image_path}")
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
    model_clip, preprocess = clip.load(CLIP_MODEL_NAME, device=device)

    pt_state = torch.load(AESTHETIC_WEIGHTS_PATH, map_location=device)
    predictor = AestheticPredictor(768)
    predictor.load_state_dict(pt_state)
    predictor.to(device)
    predictor.eval()

    return model_clip, preprocess, predictor, device

def predict_aesthetic_score(image_path, model_clip, preprocess, predictor, device):

    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model_clip.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.float()
        
        score = predictor(image_features)

    return score.item()

def get_aesthetic_score(image_path, model_clip, preprocess, predictor, device):
    score = predict_aesthetic_score(image_path, model_clip, preprocess, predictor, device)
    return score

def get_combined_score(brisque_score, aesthetic_score):
    
    aesthetic_score *= 10
    brisque_score = 100 - brisque_score

    combined_score = 0.5 * aesthetic_score + 0.5 * brisque_score
    return combined_score

def check_similarity_clip(image_path1, image_path2, model_clip, preprocess, device):
    image1 = preprocess(Image.open(image_path1)).unsqueeze(0).to(device)
    image2 = preprocess(Image.open(image_path2)).unsqueeze(0).to(device)

    with torch.no_grad():
        features1 = model_clip.encode_image(image1)
        features2 = model_clip.encode_image(image2)

        features1 = features1 / features1.norm(dim=-1, keepdim=True)
        features2 = features2 / features2.norm(dim=-1, keepdim=True)

        sim = F.cosine_similarity(features1, features2).item()
        print(f"CLIP Similarity: {sim * 100 :.4f}")
    return sim

def check_similarity_hash(image_path1, image_path2):
    hash1 = imagehash.phash(Image.open(image_path1))
    hash2 = imagehash.phash(Image.open(image_path2))

    distance = hash1 - hash2
    print(f"Hash Distance: {distance}")
    return distance

def remove_duplicates(folder_path):
    hashes = {}
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(folder_path, filename)
            img_hash = imagehash.phash(Image.open(image_path))
            if img_hash in hashes:
                print(f"Duplicate found: {filename} is a duplicate of {hashes[img_hash]}")
                os.remove(image_path)
            else:
                hashes[img_hash] = filename

def remove_similar_images(folder_path, model_clip, preprocess, device, threshold=0.9):
    features = {}
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(folder_path, filename)
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model_clip.encode_image(image)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                features[filename] = image_features

    for filename1, feat1 in features.items():
        for filename2, feat2 in features.items():
            if filename1 >= filename2:
                continue
            sim = F.cosine_similarity(feat1, feat2).item()
            if sim > threshold:
                print(f"Similar images: {filename1} and {filename2} (Similarity: {sim * 100:.4f})")
                score1 = get_brisque_score(os.path.join(folder_path, filename1))
                score2 = get_brisque_score(os.path.join(folder_path, filename2))
                if score1 is not None and score2 is not None:
                    if score1 < score2:
                        print(f"Removing {filename1} (BRISQUE: {score1:.2f} vs {score2:.2f})")
                        os.remove(os.path.join(folder_path, filename1))
                    else:
                        print(f"Removing {filename2} (BRISQUE: {score2:.2f} vs {score1:.2f})")
                        os.remove(os.path.join(folder_path, filename2))

def clasify_folder(folder_path, model_clip, preprocess, predictor, device):
    brisque_scores = {}
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(folder_path, filename)
            brisque_score = get_brisque_score(image_path)
            brisque_scores[filename] = brisque_score


    filenames_cleaned = [
        f for f in brisque_scores.keys()
        if brisque_scores[f] is not None and brisque_scores[f] < SEUIL_REJET_BRISQUE
    ]
    aesthetic_scores = {}

    for filename in filenames_cleaned:
        image_path = os.path.join(folder_path, filename)
        aesthetic_score = get_aesthetic_score(image_path, model_clip, preprocess, predictor, device)
        aesthetic_scores[filename] = aesthetic_score

    final_filenames = [f for f in aesthetic_scores.keys() if aesthetic_scores[f] > SEUIL_MINIMUM_LAION]
    final_scores = {}

    for filename in final_filenames:
        combined_score = get_combined_score(brisque_scores[filename], aesthetic_scores[filename])
        final_scores[filename] = combined_score

    return brisque_scores, aesthetic_scores, final_scores

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
    ax.set_xticklabels(filenames, rotation=45, ha='right')
    ax.set_xlabel('Image Filename')
    ax.set_ylabel('Score')
    ax.set_title('Scores for Images in Folder')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    model_clip, preprocess, predictor, device = setup_aesthetic_predictor()

    # folder_path = DATA_DIR
    # brisque_scores, aesthetic_scores, final_scores = clasify_folder(folder_path, model_clip, preprocess, predictor, device)
    # brisque_scores = dict(sorted(brisque_scores.items(), key=lambda item: item[1], reverse=True))
    # aesthetic_scores = dict(sorted(aesthetic_scores.items(), key=lambda item: item[1], reverse=True))
    # final_scores = dict(sorted(final_scores.items(), key=lambda item: item[1], reverse=True))
    
    # for filename in final_scores.keys():
    #     print(f"Selected: {filename} - Aesthetic Score: {aesthetic_scores[filename]:.2f}, BRISQUE Score: {brisque_scores[filename]:.2f}")
    # plot_scores(brisque_scores, folder_path)
    # plot_scores(aesthetic_scores, folder_path)
    # plot_scores(final_scores, folder_path)

    remove_duplicates(DATA_DIR)
    remove_similar_images(DATA_DIR, model_clip, preprocess, device, threshold=0.90)
