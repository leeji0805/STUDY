
import numpy as np
from generator_model import Generator
import dnnlib.tflib as tflib

def generate_child_face(father_latent_path, mother_latent_path, age, gender, influence, model_path, output_path):
    tflib.init_tf()
    
    # Load model
    generator = Generator(model_path, batch_size=1, randomize_noise=False)
    
    # Load latent representations
    father_latent = np.load(father_latent_path)
    mother_latent = np.load(mother_latent_path)

    # Create child latent representation
    child_latent = ((1 - influence) * father_latent + influence * mother_latent)

    # Generate child face
    child_face = generator.generate_image(child_latent)
    
    # Save the generated image
    child_face.save(output_path)
    
    print(f"Child face generated and saved to {output_path}.")
