
import os
from align_images_modified import align_images
from encode_images_modified import encode_images

def preprocess_images(input_images_dir, aligned_images_dir, generated_images_dir, latent_representations_dir, 
                      father_image_name, mother_image_name):
    # Create necessary directories
    for dir_path in [aligned_images_dir, generated_images_dir, latent_representations_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    # Align images
    align_images(input_images_dir, aligned_images_dir)
    
    # Generate latent representations
    encode_images(aligned_images_dir, generated_images_dir, latent_representations_dir)
    
    father_latent_path = os.path.join(latent_representations_dir, f"{father_image_name}.npy")
    mother_latent_path = os.path.join(latent_representations_dir, f"{mother_image_name}.npy")
    
    # Check if the latent representation files are successfully created
    if os.path.exists(father_latent_path) and os.path.exists(mother_latent_path):
        print(f"Latent representations have been successfully created:\n{father_latent_path}\n{mother_latent_path}")
        return father_latent_path, mother_latent_path
    else:
        print("Error: Unable to find the latent representation files. Please check the paths and try again.")
        return None, None
