import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import itertools
import os
import random
from tqdm import tqdm
from efficientAD.common import get_autoencoder, get_pdn_small, get_pdn_medium, \
    ImageFolderWithoutTarget, ImageFolderWithPath, InfiniteDataloader,Config
from sklearn.metrics import roc_auc_score
import torch
import itertools
import numpy as np
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import cv2
from datetime import datetime, timedelta
import time
from sklearn.metrics import f1_score
from PIL import Image
from lion_pytorch import Lion


class EfficientAD:
    def __init__(self, config):
        self.seed = config.seed
        self.image_size = config.image_size
        self.out_channels = config.out_channels
        self.on_gpu = torch.cuda.is_available()
        self.brightness = config.brightness
        self.contrast = config.contrast
        self.saturation = config.saturation

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.config = config

        self.dataset_path = self._get_dataset_path()
        self.train_output_dir = os.path.join(
            config.output_dir, config.model_name)
        
        os.makedirs(self.train_output_dir, exist_ok=True)
       
        self.should_stop = False

        self.teacher = None
        self.student = None
        self.autoencoder = None
        self.teacher_mean = None
        self.teacher_std = None
        self.q_st_start = None
        self.q_st_end = None
        self.q_ae_start = None
        self.q_ae_end = None


    def _get_dataset_path(self):
        
        return self.config.dataset_path

    def _get_data_loaders(self):
        full_train_set = ImageFolderWithoutTarget(
            self.dataset_path,
            transform=transforms.Lambda(self._train_transform))
        
        
        train_size = int(0.9 * len(full_train_set))
        validation_size = len(full_train_set) - train_size
        rng = torch.Generator().manual_seed(self.seed)
        train_set, validation_set = torch.utils.data.random_split(
            full_train_set, [train_size, validation_size], rng)
     

        train_loader = DataLoader(train_set, batch_size=self.config.batch_size, shuffle=True,
                                  num_workers=0, pin_memory=True)
        train_loader_infinite = InfiniteDataloader(train_loader)
        validation_loader = DataLoader(validation_set, batch_size=self.config.batch_size)

        return train_loader_infinite, validation_loader, train_loader

    def _train_transform(self, image):
        default_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        transform_ae = transforms.RandomChoice([
            transforms.ColorJitter(brightness=self.brightness),
            transforms.ColorJitter(contrast=self.contrast),
            transforms.ColorJitter(saturation=self.saturation),
        ])
        return default_transform(image), default_transform(transform_ae(image))

    def train(self):
        train_loader_infinite, validation_loader, train_loader = self._get_data_loaders()
        teacher, student, autoencoder = self._initialize_models()

        teacher.eval()
        student.train()
        autoencoder.train()
        final_loss = []

        if self.on_gpu:
            teacher.cuda()
            student.cuda()
            autoencoder.cuda()

        optimizer, scheduler = self._setup_optimizer_and_scheduler(student, autoencoder)

        teacher_mean, teacher_std = self._teacher_normalization(teacher, train_loader)
        
        start_time = datetime.utcnow()

        tqdm_obj = tqdm(range(self.config.train_steps))
        for iteration, (image_st, image_ae), _ in zip(tqdm_obj, train_loader_infinite, itertools.repeat(None)):
            
            if self.should_stop:  
                print("Training stopped by user. Saving model...")
                break
            
            if self.on_gpu:
                image_st, image_ae = image_st.cuda(), image_ae.cuda()
            
            with torch.no_grad():
                teacher_output_st = teacher(image_st)
                teacher_output_st = (teacher_output_st - teacher_mean) / teacher_std

            student_output_st = student(image_st)[:, :self.out_channels]
            distance_st = (teacher_output_st - student_output_st) ** 2
            d_hard = torch.quantile(distance_st, q=0.999)
            loss_hard = torch.mean(distance_st[distance_st >= d_hard])
            loss_st = loss_hard
            
            ae_output = autoencoder(image_ae)
            with torch.no_grad():
                teacher_output_ae = teacher(image_ae)
                teacher_output_ae = (teacher_output_ae - teacher_mean) / teacher_std
            student_output_ae = student(image_ae)[:, self.out_channels:]
            distance_ae = (teacher_output_ae - ae_output)**2
            distance_stae = (ae_output - student_output_ae)**2
            loss_ae = torch.mean(distance_ae)
            loss_stae = torch.mean(distance_stae)
            loss_total = loss_st + loss_ae + loss_stae

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            scheduler.step()
            final_loss.append(loss_total.item())

            if iteration % 10 == 0:
                tqdm_obj.set_description(f"Current loss: {loss_total.item():.4f}")

            if iteration % 10000 == 0 and iteration > 0:
                teacher.eval()
                student.eval()
                autoencoder.eval()

                q_st_start, q_st_end, q_ae_start, q_ae_end = self.map_normalization(
                    validation_loader=validation_loader, teacher=teacher,
                    student=student, autoencoder=autoencoder,
                    teacher_mean=teacher_mean, teacher_std=teacher_std,
                    desc='Intermediate map normalization')
                
           
                teacher.eval()
                student.train()
                autoencoder.train()

        
        teacher.eval()
        student.eval()
        autoencoder.eval()


        q_st_start, q_st_end, q_ae_start, q_ae_end = self.map_normalization(
            validation_loader=validation_loader, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std, desc='Final map normalization')
        
        self.teacher = teacher
        self.student = student
        self.autoencoder = autoencoder
        self.teacher_mean = teacher_mean
        self.teacher_std = teacher_std
        self.q_st_start = q_st_start
        self.q_st_end = q_st_end
        self.q_ae_start = q_ae_start
        self.q_ae_end = q_ae_end
        

        model_save_path = self._save_checkpoint(teacher, student, autoencoder,teacher_mean, teacher_std, q_st_start, q_st_end, q_ae_start, q_ae_end)
        end_time = datetime.utcnow()  
        training_duration = (end_time - start_time).total_seconds()  

        training_time_str = str(timedelta(seconds=int(training_duration)))
        print('Finish')
        return final_loss, model_save_path, training_time_str

    def _initialize_models(self):
        if self.config.model_size == 'small':
            teacher = get_pdn_small(self.out_channels)
            student = get_pdn_small(2 * self.out_channels)
        elif self.config.model_size == 'medium':
            teacher = get_pdn_medium(self.out_channels)
            student = get_pdn_medium(2 * self.out_channels)
        else:
            raise ValueError("Invalid model size")

        teacher.load_state_dict(torch.load(self.config.weights, map_location='cpu'))
        autoencoder = get_autoencoder(self.out_channels, self.config.image_size)

        return teacher, student, autoencoder


    def _setup_optimizer_and_scheduler(self, student, autoencoder):
        # optimizer = torch.optim.Adam(
        #     itertools.chain(student.parameters(), autoencoder.parameters()),
        #     lr=1e-4, weight_decay=1e-5)
        optimizer = Lion(
            itertools.chain(student.parameters(), autoencoder.parameters()),
            lr=1e-4, weight_decay=1e-5)
      

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.train_steps, eta_min=0
        )
        return optimizer, scheduler


    
    @torch.no_grad()
    def _teacher_normalization(self, teacher, train_loader):
        mean_outputs = []
        
        for train_image, _ in tqdm(train_loader, desc='Computing mean of features'):
            if self.on_gpu:
                train_image = train_image.cuda()
            
            teacher_output = teacher(train_image)
            
            mean_output = torch.mean(teacher_output, dim=[0, 2, 3])
            mean_outputs.append(mean_output)  # Move to CPU to save memory
        
        channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)
        channel_mean = channel_mean[None, :, None, None]

        mean_distances = []
        for train_image, _ in tqdm(train_loader, desc='Computing std of features'):
            if self.on_gpu:
                train_image = train_image.cuda()
          
            teacher_output = teacher(train_image)
            distance = (teacher_output - channel_mean) ** 2
            mean_distance = torch.mean(distance, dim=[0, 2, 3])
            mean_distances.append(mean_distance)  # Move to CPU
        channel_var = torch.mean(torch.stack(mean_distances), dim=0)
        channel_var = channel_var[None, :, None, None]
        channel_std = torch.sqrt(channel_var)

        return channel_mean, channel_std


  

    def _save_checkpoint(self, teacher, student, autoencoder,teacher_mean, teacher_std, q_st_start, q_st_end, q_ae_start, q_ae_end):
        

        model_save_path = os.path.join(self.train_output_dir, f'{self.config.model_name}.pth')
        torch.save({
            'teacher': teacher,
            'student': student,
            'autoencoder': autoencoder,
            'teacher_mean': teacher_mean,
            'teacher_std': teacher_std,
            'q_st_start': q_st_start,
            'q_st_end': q_st_end,
            'q_ae_start': q_ae_start,
            'q_ae_end': q_ae_end,
        }, model_save_path)

        return model_save_path



    @torch.no_grad()
    def map_normalization(self,validation_loader, teacher,student, autoencoder,teacher_mean, teacher_std,desc='Intermediate map normalization'):
        maps_st = []
        maps_ae = []
        for image, _ in tqdm(validation_loader, desc=desc):
            if self.on_gpu:
                image = image.cuda()
            
            map_combined, map_st, map_ae = self.predict(
                image=image, teacher=teacher, student=student,
                autoencoder=autoencoder, teacher_mean=teacher_mean,
                teacher_std=teacher_std)
            maps_st.append(map_st)
            maps_ae.append(map_ae)
        maps_st = torch.cat(maps_st)
        maps_ae = torch.cat(maps_ae)
        q_st_start = torch.quantile(maps_st, q=0.9)
        q_st_end = torch.quantile(maps_st, q=0.995)
        q_ae_start = torch.quantile(maps_ae, q=0.9)
        q_ae_end = torch.quantile(maps_ae, q=0.995)
        return q_st_start, q_st_end, q_ae_start, q_ae_end
    

   

    @torch.no_grad()
    def predict(self,image, teacher, student, autoencoder, teacher_mean, teacher_std,
                q_st_start=None, q_st_end=None, q_ae_start=None, q_ae_end=None):
        teacher_output = teacher(image)
        teacher_output = (teacher_output - teacher_mean) / teacher_std
        student_output = student(image)
        autoencoder_output = autoencoder(image)
        map_st = torch.mean((teacher_output - student_output[:, :self.out_channels])**2,
                            dim=1, keepdim=True)
  
        map_ae = torch.mean((autoencoder_output -
                            student_output[:, self.out_channels:])**2,
                            dim=1, keepdim=True)
        if q_st_start is not None:
            map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
        if q_ae_start is not None:
            map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)
        map_combined = 0.5 * map_st + 0.5 * map_ae
        return map_combined, map_st, map_ae


    @torch.no_grad()
    def predict_one_image(self, image, teacher=None, student=None, autoencoder=None, 
                teacher_mean=None, teacher_std=None):
        
        if teacher is None:
            teacher = self.teacher
        if student is None:
            student = self.student
        if autoencoder is None:
            autoencoder = self.autoencoder
        if teacher_mean is None:
            teacher_mean = self.teacher_mean
        if teacher_std is None:
            teacher_std = self.teacher_std
        
        orig_width = image.width
        orig_height = image.height
        image, _ = self._train_transform(image)
        if self.on_gpu:
            image = image.cuda()
        image = image.unsqueeze(0)
        
        teacher_output = teacher(image)
        teacher_output = (teacher_output - teacher_mean) / teacher_std
        student_output = student(image)
       
        autoencoder_output = autoencoder(image)
        out_channels = student_output.size(1) // 2  
        map_st = torch.mean((teacher_output - student_output[:, :out_channels])**2,
                            dim=1, keepdim=True)
        map_ae = torch.mean((autoencoder_output - 
                            student_output[:, out_channels:])**2,
                            dim=1, keepdim=True)
      
        if self.q_st_start is not None:
            map_st = 0.1 * (map_st - self.q_st_start) / (self.q_st_end - self.q_st_start)
        if self.q_ae_start is not None:
            map_ae = 0.1 * (map_ae - self.q_ae_start) / (self.q_ae_end - self.q_ae_start)

        map_combined = 0.5 * map_st + 0.5 * map_ae

        map_combined = torch.nn.functional.pad(map_combined, (1, 1, 1, 1))
        map_combined = torch.nn.functional.interpolate(
            map_combined, (orig_height, orig_width), mode='bilinear')
        map_combined = map_combined[0, 0].cpu().numpy()
        
        y_score_image = np.max(map_combined)
        
        return y_score_image, map_combined
    

    def initialize_predict(self):

        pth_file = torch.load(self.config.pretrain_model_path)

        self.teacher = pth_file["teacher"]
        self.student = pth_file["student"]
        self.autoencoder = pth_file["autoencoder"]
        self.teacher_mean = pth_file["teacher_mean"]
        self.teacher_std = pth_file["teacher_std"]
        self.q_st_start = pth_file["q_st_start"]
        self.q_st_end = pth_file["q_st_end"]
        self.q_ae_start = pth_file["q_ae_start"]
        self.q_ae_end = pth_file["q_ae_end"]

        if self.on_gpu:
            self.teacher.cuda()
            self.student.cuda()
            self.autoencoder.cuda()
            self.teacher_mean.cuda()
            self.teacher_std.cuda()
            self.q_st_start.cuda()
            self.q_st_end.cuda()
            self.q_ae_start.cuda()
            self.q_ae_end.cuda()
        

        
    def stop_training(self):
        self.should_stop = True

    