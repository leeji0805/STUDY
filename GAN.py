
import os
import cv2
import math
import pickle
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFilter
import dnnlib
import matplotlib.pyplot as plt
from encoder.generator_model import Generator
import dnnlib.tflib as tflib

def load_latent_directions():
    directions = {
        'age': np.load('age.npy'),
        'gender': np.load('gender.npy')
    }
    return directions

def generate_final_image(latent_vector, generator):
    latent_vector = latent_vector.reshape((1, 18, 512))
    generator.set_dlatents(latent_vector)
    img_array = generator.generate_images()[0]
    img = Image.fromarray(img_array, 'RGB')
    return img

def generate_child_image(father_latent_path, mother_latent_path, age_input, gender_input, model_path='karras2019stylegan-ffhq-1024x1024.pkl'):
    tflib.init_tf()
    tf.compat.v1.disable_eager_execution()
    _G, _D, Gs = pickle.load(open(model_path, 'rb'))
    generator = Generator(Gs, batch_size=1, randomize_noise=False)
    
    father_latent = np.load(father_latent_path)
    mother_latent = np.load(mother_latent_path)

    latent_directions = load_latent_directions()

    def map_age_coeff_to_age(age_coeff):
        m = 6.33  # slope
        b = 26   # intercept
        age = m * age_coeff + b
        return round(age)

    age_coeff = -((age_input / 5) - 6)
    age_coeff = map_age_coeff_to_age(age_coeff)
        
    gender_intensity = 1 if gender_input == 'male' else -1
        
    genes_influence = 0.5
    final_latent = ((1 - genes_influence) * father_latent + genes_influence * mother_latent)
    final_latent[:8] += age_coeff * latent_directions['age'][:8]
    final_latent[:8] += gender_intensity * latent_directions['gender'][:8]

    final_image = generate_final_image(final_latent, generator)
    final_image_path = 'generated_images/final_child.png'
    final_image.save(final_image_path)
    return final_image_path
