
import os

# Local imports
from preprocessing_and_encoding import preprocess_and_encode
from prepro import preprocess_images
from GAN import generate_child_image

def main():
    # Directories and paths
    input_dir = "testpoto"
    aligned_dir = "D:/SSLHK/aligned_images/"
    latent_dir = "D:/SSLHK/latent_representations/"
    # model_path = "model/karras2019stylegan-ffhq-1024x1024.pkl"
    output_path = "D:/SSLHK/generated_images"
    for dir_path in [aligned_dir, output_path, latent_dir]:
        if not os.path.exists(dir_path):
         os.makedirs(dir_path)

    # # Get inputs
    # age = input("Enter age: ")
    # gender = input("Enter gender (male/female): ")
    # influence = float(input("Enter genetic influence (0 to 1): "))
    
    # Preprocess and encode images
    preprocess_images(input_dir, aligned_dir, output_path, latent_dir, "D:/SSLHK/testpoto/father.jpg", "D:/SSLHK/testpoto/mother.jpg")

    # # Generate child face
    # father_latent_path = os.path.join(latent_dir, "father.npy")
    # mother_latent_path = os.path.join(latent_dir, "mother.npy")
    
    # generate_child_image(father_latent_path, mother_latent_path, age, gender, influence, model_path, output_path)

if __name__ == "__main__":
    main()
