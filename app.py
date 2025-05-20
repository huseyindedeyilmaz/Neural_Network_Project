from flask import Flask, render_template, request
import os
import numpy as np
from PIL import Image
import joblib
from efficientAD.efficientADModel import EfficientAD
from efficientAD.common import Config
from glass_model.glass import GLASS
from anomalib.deploy import OpenVINOInferencer
import torch
import glass_model.backbones as backbones
import torchvision.transforms as transforms
from anomalib.deploy import TorchInferencer
import pathlib
import sys
from datetime import datetime
import cv2


app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Glass

device = "cuda" if torch.cuda.is_available() else "cpu"

glass_path = "./models/glass/ckpt_best_5.pth"

checkpoint = torch.load(glass_path, weights_only=True)
backbone = backbones.load("wideresnet50")

glass = GLASS(device)
glass.load(
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


# Padim



pathlib.PosixPath = pathlib.WindowsPath

padim_path = "./models/padim/model.pt"

padim = TorchInferencer(
    path=padim_path,
    device="cuda",
)

# EfficientAD
config = Config(pretrain_model_path="./models/efficientad/model/model.pth",model_name="model")

efficientad = EfficientAD(config=config)
efficientad.initialize_predict()






@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = {}
    image_url = None


    if request.method == 'POST':
        file = request.files['image']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if file:
            filename = file.filename
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            image_url = filepath

            image = Image.open(filepath).convert('RGB')
            np_img = np.array(image)

            # PaDiM ile tahmin
            try:
                padim_pred = padim.predict(image)
                print(padim_pred.anomaly_map[0].shape)
                padim_label = "Anomaly" if padim_pred.pred_score > 0.5 else "Normal"
                

                padim_mask = padim_pred.anomaly_map[0].cpu().numpy() 
                norm_mask = (padim_mask - padim_mask.min()) / (padim_mask.max() - padim_mask.min() + 1e-10)
                norm_mask = (norm_mask * 255).astype("uint8")
                norm_mask = cv2.resize(norm_mask, image.size)
                colored_padim_mask = cv2.applyColorMap(norm_mask, cv2.COLORMAP_JET)

                mask_filename = f"padim_mask_{timestamp}.png"
                mask_path = os.path.join(UPLOAD_FOLDER, mask_filename)
                cv2.imwrite(mask_path, colored_padim_mask)

                predictions["PaDiM"] = f"{padim_label} (Score: {padim_pred.pred_score[0][0]})"
                predictions["PaDiM_mask_url"] = f"uploads/{mask_filename}"

            except Exception as e:
                predictions["PaDiM"] = f"Hata: {str(e)}"

            # GLASS
            try:
                img = transform_img(image).unsqueeze(0).to(device)
                score, mask_glass = glass._predict(img)
                label = "Anomaly" if score[0] > 0.9447653 else "Normal"

                img_shape = img.shape[2:]  # (H, W)
                mask_array = np.array(mask_glass).squeeze()
                min_val = mask_array.min()
                max_val = mask_array.max()
                norm_mask = (mask_array - min_val) / (max_val - min_val + 1e-10)
                norm_mask = cv2.resize(norm_mask, img_shape[::-1])
                norm_mask = (norm_mask * 255).astype("uint8")
                colored_mask_glass = cv2.applyColorMap(norm_mask, cv2.COLORMAP_JET)

                mask_filename = f"glass_mask_{timestamp}.png"
                mask_path = os.path.join(UPLOAD_FOLDER, mask_filename)
                cv2.imwrite(mask_path, colored_mask_glass)

                predictions["GLASS_mask_url"] = f"uploads/{mask_filename}"

                predictions["GLASS"] = f"{label} (Score: {score[0]:.4f})"
            except Exception as e:
                predictions["GLASS"] = f"Hata: {str(e)}"

            # EfficientAD
            try:
                score, mask_efficient_ad = efficientad.predict_one_image(image)
                label = "Anomaly" if score > 0.24 else "Normal"

                norm_mask = (mask_efficient_ad - mask_efficient_ad.min()) / (mask_efficient_ad.max() - mask_efficient_ad.min() + 1e-10)
                norm_mask = (norm_mask * 255).astype("uint8")
                norm_mask = cv2.resize(norm_mask, image.size)
                colored_mask_efficient_ad = cv2.applyColorMap(norm_mask, cv2.COLORMAP_JET)

                mask_filename = f"efficientad_mask_{timestamp}.png"
                mask_path = os.path.join(UPLOAD_FOLDER, mask_filename)
                cv2.imwrite(mask_path, colored_mask_efficient_ad)

                predictions["EfficientAD"] = f"{label} (Score: {score:.4f})"
                predictions["EfficientAD_mask_url"] = f"uploads/{mask_filename}"
            except Exception as e:
                predictions["EfficientAD"] = f"Hata: {str(e)}"

    return render_template("index.html", predictions=predictions, image_url=image_url)

if __name__ == '__main__':
    app.run(debug=True)
