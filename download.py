# This file runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model
import os

def download_model():
    os.makedirs('gfpgan/weights', exist_ok=True)
    # download weights
    if not os.path.exists('gfpgan/weights/realesr-general-x4v3.pth'):
        os.system(
            'wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth -P ./gfpgan/weights'
        )
    if not os.path.exists('gfpgan/weights/GFPGANv1.2.pth'):
        os.system(
            'wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.2.pth -P ./gfpgan/weights')
    if not os.path.exists('gfpgan/weights/GFPGANv1.3.pth'):
        os.system(
            'wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth -P ./gfpgan/weights')
    if not os.path.exists('gfpgan/weights/GFPGANv1.4.pth'):
        os.system(
            'wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth -P ./gfpgan/weights')
    if not os.path.exists('gfpgan/weights/RestoreFormer.pth'):
        os.system(
            'wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth -P ./gfpgan/weights'
        )
    

if __name__ == "__main__":
    download_model()