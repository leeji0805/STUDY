import os
import pickle
import numpy as np
import tensorflow as tf
from PIL import Image
import sys
sys.path.append('D:/SSLHK')
import dnnlib
from encoder.generator_model import Generator
import dnnlib.tflib as tflib


def load_latent_directions():
    directions = {
        'age': np.load('D:/SSLHK/ffhq_dataset/latent_directions/age.npy'),
        'gender': np.load('D:/SSLHK/ffhq_dataset/latent_directions/gender.npy')
    }
    return directions


def initialize_generator(model_path):
    tflib.init_tf()
    tf.compat.v1.disable_eager_execution()
    _G, _D, Gs = pickle.load(open(model_path, 'rb'))
    return Generator(Gs, batch_size=1, randomize_noise=False)


def generate_final_image(latent_vector, generator):
    latent_vector = latent_vector.reshape((1, 18, 512))
    generator.set_dlatents(latent_vector)
    img_array = generator.generate_images()[0]
    img = Image.fromarray(img_array, 'RGB')
    return img


def create_child(father_latent, mother_latent, age_input, gender_input, influence_input, latent_directions):
    age_coeff = (age_input - 26) / 6.33
    age_coeff = max(min(age_coeff, 3), -3)
    gender_intensity = 1 if gender_input == 'male' else -1
    genes_influence = influence_input
    final_latent = ((1 - genes_influence) * father_latent + genes_influence * mother_latent)
    final_latent[:8] += age_coeff * latent_directions['age'][:8]
    final_latent[:8] += gender_intensity * latent_directions['gender'][:8]
    return final_latent


def main():
    model_path = 'D:/SSLHK/model/karras2019stylegan-ffhq-1024x1024.pkl'
    generator = initialize_generator(model_path)
    
    father_latent_path = 'D:/SSLHK/latent_representations/BJ_01.npy'
    mother_latent_path = 'D:/SSLHK/latent_representations/GY_01.npy'
    
    father_latent = np.load(father_latent_path)
    mother_latent = np.load(mother_latent_path)
    
    latent_directions = load_latent_directions()

    while True:
        age_input = input("Please enter the age of the child or 'exit' to quit: ")
        if age_input.lower() == 'exit':
            break
        try:
            age_input = int(age_input)
        except ValueError:
            print("Please enter a valid number for age or 'exit' to quit.")
            continue
        
        gender_input = input("Please enter the gender of the child (male/female) or 'exit' to quit: ").lower()
        if gender_input == 'exit':
            break
        elif gender_input not in ['male', 'female']:
            print("Please enter 'male', 'female' or 'exit' to quit.")
            continue
        influence_input = input("Please enter the influence of the child(0.1~0.9) or 'exit' to quit: ").lower()
        if influence_input == 'exit':
            break
        try:
            influence_input = float (influence_input)
        except ValueError:
            print("Please enter a valid number for age or 'exit' to quit.")
            continue
        
        final_latent = create_child(father_latent, mother_latent, age_input, gender_input, influence_input, latent_directions)
        
        final_image = generate_final_image(final_latent, generator)
        final_image_path = 'D:/SSLHK/generated_images/final_child.png'
        final_image.save(final_image_path)
        
        print(f"Image has been saved to {final_image_path}")

if __name__ == "__main__":
    main()
