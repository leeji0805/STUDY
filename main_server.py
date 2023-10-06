
import os

# Local imports
from preprocessing_and_encoding import preprocess_and_encode
from child_face_generator import generate_child_face

def main():
    # Directories and paths
    input_dir = "path_to_input_images"
    aligned_dir = "path_to_aligned_images"
    latent_dir = "path_to_latent_representations"
    model_path = "path_to_stylegan_model"
    output_path = "path_to_save_generated_face"

    # Get inputs
    age = input("Enter age: ")
    gender = input("Enter gender (male/female): ")
    influence = float(input("Enter genetic influence (0 to 1): "))

    # Preprocess and encode images
    preprocess_and_encode(input_dir, aligned_dir, latent_dir)

    # Generate child face
    father_latent_path = os.path.join(latent_dir, "father.npy")
    mother_latent_path = os.path.join(latent_dir, "mother.npy")
    
    generate_child_face(father_latent_path, mother_latent_path, age, gender, influence, model_path, output_path)

if __name__ == "__main__":
    main()
