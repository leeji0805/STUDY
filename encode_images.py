
import os
import pickle
from encoder.perceptual_model import PerceptualModel
import dnnlib
import dnnlib.tflib as tflib
from encoder.generator_model import Generator

def encode_images(data_dir, generated_images_dir, latent_representations_dir, load_resnet, lr, iterations):
    tflib.init_tf()
    os.makedirs(latent_representations_dir, exist_ok=True)
    
    # Initialize generator and perceptual model
    with open('stylegan_ffhq.pkl', "rb") as f:
        _, _, Gs = pickle.load(f)
    
    generator = Generator(Gs, batch_size=1, randomize_noise=False)
    perceptual_model = PerceptualModel(img_size=256, lr=lr, resnet_image_size=256)
    perceptual_model.build_perceptual_model(generator.generated_image)
    
    perceptual_model.load_weights(load_resnet)
    
    # Optimize (only) dlatents by minimizing perceptual loss between generated images and target images
    for img_name in os.listdir(data_dir):
        print('Loading %s ...' % img_name)
        target_image = perceptual_model.get_target_image(os.path.join(data_dir, img_name))
        perceptual_model.optimize_dlatents(iterations=iterations)
        generator.set_dlatents(perceptual_model.dlatents)
        
        img_array = generator.generate_images()[0]
        img = Image.fromarray(img_array, 'RGB')
        img.save(os.path.join(generated_images_dir, img_name))
        np.save(os.path.join(latent_representations_dir, img_name + '.npy'), perceptual_model.dlatents)
