import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from sklearn.metrics import f1_score
import cv2
from glass_model.glass import GLASS
import glass_model.backbones as backbones



def calculateIoU(gtMask, predMask):
    tp = np.logical_and(gtMask == 1, predMask == 1).sum()
    fp = np.logical_and(gtMask == 0, predMask == 1).sum()
    fn = np.logical_and(gtMask == 1, predMask == 0).sum()

    denominator = tp + fp + fn
    if denominator == 0:
        return 1.0  
    return tp / denominator


def evaluate_iou(model, defect_dir, gt_dir, threshold_segmentation):
    iou_scores = []
    
    defect_fnames = sorted(os.listdir(defect_dir))
    gt_fnames = sorted(os.listdir(gt_dir))
    
    for defect_fname, gt_fname in zip(defect_fnames, gt_fnames):
        if defect_fname.endswith(('.png', '.jpg', '.tiff')):
            img_path = os.path.join(defect_dir, defect_fname)
            gt_path = os.path.join(gt_dir, gt_fname)
            
            img_pil = Image.open(img_path).convert('RGB')
            img_tensor = transform_img(img_pil).unsqueeze(0).to(device)

            _, heatmap = model._predict(img_tensor)
            heatmap = np.array(heatmap).squeeze()

            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-10)
            heatmap_resized = cv2.resize(heatmap, img_pil.size)  # (W, H)

            pred_mask = (heatmap_resized > threshold_segmentation).astype(np.uint8)

            gt_mask = np.array(Image.open(gt_path).convert('L'))
            gt_mask = cv2.resize(gt_mask, img_pil.size)
            gt_mask = (gt_mask > 127).astype(np.uint8)

            iou = calculateIoU(gt_mask, pred_mask)
            iou_scores.append(iou)

    mean_iou = np.mean(iou_scores)
    print(f"Mean IoU Score: {mean_iou:.4f}")
    return mean_iou



device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint_path = r"models\glass\ckpt_best_5.pth"
dataset_path = r"wood_dataset\wood"

checkpoint = torch.load(checkpoint_path, weights_only=True)
backbone = backbones.load("wideresnet50")

model = GLASS(device)
model.load(
    backbone=backbone,
    layers_to_extract_from=["layer2", "layer3"],
    input_shape=(3, 256, 256),
    pretrain_embed_dimension=1536,
    target_embed_dimension=1536,
    device=device,
    checkpoint=checkpoint,
    dsc_hidden=1024,
    dsc_layers=2
)

IMAGENET_MEAN = [0.68372813, 0.58139985, 0.389138]
IMAGENET_STD = [0.25897288, 0.21349465, 0.14381898]

transform_img = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

train_dir = os.path.join(dataset_path, "train/good")
train_scores = []

for file in os.listdir(train_dir):
    img_path = os.path.join(train_dir, file)
    if img_path.endswith((".png", ".jpg", ".jpeg")):
        img = Image.open(img_path).convert("RGB")
        img = transform_img(img).unsqueeze(0).to(device)
        score, _ = model._predict(img)
        train_scores.append(score[0])

threshold = max(train_scores)
print("Threshold: ", threshold)

test_defect_dir = os.path.join(dataset_path, "test/defect")
test_good_dir = os.path.join(dataset_path, "test/good")

y_true = []
y_pred = []

def process_directory(path, label):
    for file in os.listdir(path):
        img_path = os.path.join(path, file)
        if img_path.endswith((".png", ".jpg", ".jpeg")):
            img = Image.open(img_path).convert("RGB")
            img = transform_img(img).unsqueeze(0).to(device)
            score, _ = model._predict(img)
            y_true.append(label)
            y_pred.append(1 if score[0] > threshold else 0)

process_directory(test_defect_dir, 1)
process_directory(test_good_dir, 0)

f1 = f1_score(y_true, y_pred)
print("F1 Score: ", f1)


iou = evaluate_iou(model, r"wood_dataset\wood\test\defect", r"wood_dataset\wood\ground_truth\defect",threshold)
print("iou :", iou)