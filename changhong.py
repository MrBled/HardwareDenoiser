# ----------------------------------------------------------------------------
# Emerald Video Denoise Accelerator
# Benchmarking script for evaluating model performance
# In this script, we benchmark the 2 models with pixel shuffle and without pixel shuffle.
# ----------------------------------------------------------------------------
import os
import cv2
import numpy as np
from pathlib import Path
import torch
import shutil
from glob import glob
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from my_models import UnetGenerator_hardware, UnetGenerator_hardware_pixelshuffle
from torchvision.transforms import ToTensor
import math
# ----------------------------------------------------------------------------
# from models.my_models_0731 import UnetGenerator_hardware_pixelshuffle as Unet_P
# from models.my_models_0731 import UnetGenerator_hardware as Unet
# # from utils.dataloader import get_dataloader
# from utils.preprocessing import process_images, process_images_grayscale
# from utils.dataloader import create_dataloader
# from utils.testmodel import print_test_results
# from utils.performance import measure_model_performance, print_performance_results



# model_path = 'models/unet_f32_0808_gray_oldtorch.pt'


def pad_to_multiple(img_tensor, multiple=16):
    """Pad a 4D tensor (B, C, H, W) so that H and W are divisible by `multiple`."""
    _, _, h, w = img_tensor.shape
    new_h = math.ceil(h / multiple) * multiple
    new_w = math.ceil(w / multiple) * multiple
    pad_h = new_h - h
    pad_w = new_w - w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    padding = (pad_left, pad_right, pad_top, pad_bottom)  # (left, right, top, bottom)

    return torch.nn.functional.pad(img_tensor, padding, mode='reflect'), padding

model_path = "/data/clement/models/hw_denoiser_grayscale_continue21_10_58/model_epoch_9940.pt"
# model_path = "/data/clement/models/hw_denoiser_grayscale_continue21_10_58/model_epoch_980.pt"

test_figure_path = "/home/bledc/Pictures/CBSD68-dataset/CBSD68/grayscale_noise15/0000.png"
test_figure_path = "set12_sample_gray.png" 
test_result_path = './test_results'


def main():
    

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model = UnetGenerator_hardware(1, 1, 8).to(device)
    # model = Unet(input_nc=1, output_nc=1, num_downs=8)
    model.load_state_dict(torch.load(model_path)["model_state_dict"], strict=True)
    model.eval()

    # check if all the keys match
    model_keys = set(model.state_dict().keys())
    loaded_keys = set(torch.load(model_path).keys())
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Create output directory if it doesn't exist
    if not os.path.exists(test_result_path):
        os.makedirs(test_result_path)
    # input the image into the model
    # check test image is grayscale
    # if len(cv2.imread(test_figure_path, cv2.IMREAD_GRAYSCALE).shape) != 2:
    #     print("Test image is not grayscale. Converting to grayscale.")
    #     test_image = cv2.imread(test_figure_path)
    #     test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    # else:
    #     print("Test image is grayscale.")
    
    test_image = cv2.imread(test_figure_path, cv2.IMREAD_GRAYSCALE)

    test_image_tensor = ToTensor()(test_image).unsqueeze(0).to(device)
    # test_image_tensor = (ToTensor()(test_image) * 2 - 1)  # [0,1] -> [-1,1]
    # test_image_tensor = test_image_tensor.unsqueeze(0).to(device)
    # test_image_tensor, _ = pad_to_multiple(test_image_tensor, multiple=512)
    
    with torch.no_grad():
        output_image_tensor = model(test_image_tensor)
        
    # Convert output tensor to image
    output_image = output_image_tensor.squeeze().cpu().numpy()
    output_image = (output_image * 255).astype(np.uint8)

    # output_image = output_image_tensor.squeeze().cpu().numpy()
    # output_image = (output_image + 1) * 127.5  # [-1,1] -> [0,255]
    # output_image = np.clip(output_image, 0, 255).astype(np.uint8)
    # Test output image is grayscale
    if len(output_image.shape) != 2:
        print("Output image is not grayscale. Converting to grayscale.")
        output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
    else:
        print("Output image is grayscale.")

    cv2.imwrite(os.path.join(test_result_path, "output_unpadded.png"), output_image)
    print(f"Output image saved to {test_result_path}/output.png")

if __name__ == "__main__":
    main()