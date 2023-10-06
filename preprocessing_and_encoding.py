
import os
from align_images import align_images
from encode_images import encode_images

def preprocess_and_encode(input_dir, aligned_dir, latent_dir, model_res=256):
    # Align images
    align_images(input_dir, aligned_dir, model_res=model_res)

    # Encode images
    encode_images(aligned_dir, latent_dir)

    print("Preprocessing and encoding completed.")
