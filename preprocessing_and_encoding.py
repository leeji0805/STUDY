
import os
import subprocess

def preprocess_and_encode(input_images_dir, aligned_images_dir, generated_images_dir, latent_representations_dir, 
                      father_image_name, mother_image_name):
    for dir_path in [aligned_images_dir, generated_images_dir, latent_representations_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    python_path = 'd:\\SSLHK\\GAN\\Scripts\\python.exe'        
    # Align images
    subprocess.run([python_path, "align_images.py", input_images_dir, aligned_images_dir])

    # Encode images
    subprocess.run([python_path, "encode_images.py", aligned_images_dir, generated_images_dir, latent_representations_dir])

    father_latent_path = os.path.join(latent_representations_dir, f"{father_image_name}.npy")
    mother_latent_path = os.path.join(latent_representations_dir, f"{mother_image_name}.npy")
    
    # Check if the latent representation files are successfully created
    if os.path.exists(father_latent_path) and os.path.exists(mother_latent_path):
        print(f"Latent representations have been successfully created:\n{father_latent_path}\n{mother_latent_path}")
        return father_latent_path, mother_latent_path
    else:
        print("Error: Unable to find the latent representation files. Please check the paths and try again.")
        return None, None
