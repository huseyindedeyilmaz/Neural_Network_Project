from glass_model import GLASS
from model import Discriminator, Projection
import torch
import torchvision.transforms as transforms
import backbones
import PIL
import numpy as np
import cv2
import os

IMAGENET_MEAN = [0.68372813, 0.58139985, 0.389138]
IMAGENET_STD = [0.25897288, 0.21349465, 0.14381898]

device = "cuda" if torch.cuda.is_available() else "cpu"

# Model ağırlıklarını yükleme
checkpoint = torch.load('./model_weights/ckpt_best_18.pth', weights_only=True)

backbone = backbones.load("wideresnet50")
layers_to_extract_from = ["layer2", "layer3"]
input_shape = (3, 512, 512)
pretrain_embed_dimension = 1800
target_embed_dimension = 1800

# Model yükleme
model = GLASS(device)
model.load(
    backbone=backbone,
    layers_to_extract_from=layers_to_extract_from,
    input_shape=input_shape,
    pretrain_embed_dimension=pretrain_embed_dimension,
    target_embed_dimension=target_embed_dimension,
    device=device,
    checkpoint=checkpoint,
    dsc_hidden = 1024,
    dsc_layers = 3
)

# Resim dönüşüm işlemleri
resize = (512, 512)
transform_img = [
            transforms.Resize(512),
            transforms.ColorJitter(0, 0, 0),
            transforms.RandomHorizontalFlip(0),
            transforms.RandomVerticalFlip(0),
            transforms.RandomGrayscale(0),
            transforms.RandomAffine(0,
                                    translate=(0, 0),
                                    scale=(1.0 - 0, 1.0 + 0),
                                    interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
transform_img = transforms.Compose(transform_img)


#image_path = "C:/Users/hy138/Desktop/resize/wood/100000002.jpg"
#image_path = "./wood_512_512/100000003.jpg"
image_path = "./wood/100000000.jpg"
image = PIL.Image.open(image_path).convert('RGB') 
image.save('./original_image.png')  

image_tensor = transform_img(image).unsqueeze(0).to(device)

transform_image_pil = transforms.ToPILImage()(image_tensor.squeeze(0)) 
transform_image_pil.save('./transformed_image.jpg')  # 
print("Transformed image saved as 'transformed_image.jpg'")

score, mask = model._predict(image_tensor)

mask_array = np.array(mask).squeeze()

min_scores = mask_array.min()
max_scores = mask_array.max()
print(score[0])
norm_segmentations = (mask_array - min_scores) / (max_scores - min_scores + 1e-10)

grayscale_mask = cv2.cvtColor(cv2.resize(norm_segmentations, (512, 512)), cv2.COLOR_GRAY2BGR)  # Resize işlemi
grayscale_mask = (grayscale_mask * 255).astype('uint8')
# print(grayscale_mask)
# print(grayscale_mask.shape,"+++")


grayscale_mask = grayscale_mask.squeeze()

_, thresholded_mask = cv2.threshold(grayscale_mask, 150, 255, cv2.THRESH_BINARY)
# print(thresholded_mask.shape)

PIL.Image.fromarray(thresholded_mask).save("mask_grayscale.jpg")
print("Grayscale mask saved as 'mask_grayscale.jpg'")

colored_mask = cv2.applyColorMap(grayscale_mask, cv2.COLORMAP_JET)
cv2.imwrite("mask_colored.jpg", colored_mask)
print("Colored mask saved as 'mask_colored.jpg'")




















