import os
import subprocess
from align_images import align
# 경로 설정
input_images_dir = "D:/SSLHK/testpoto/"
aligned_images_dir = "D:/SSLHK/aligned_images/"
generated_images_dir = "D:/SSLHK/generated_images/"
latent_representations_dir = "D:/SSLHK/latent_representations/"

# 필요한 디렉토리 생성
for dir_path in [aligned_images_dir, generated_images_dir, latent_representations_dir]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# 이미지 정렬
align(input_images_dir, aligned_images_dir)

# latent representation 생성
subprocess.run(["D:/SSLHK/GAN/Scripts/python.exe", "D:/SSLHK/encode_images.py", aligned_images_dir, generated_images_dir, latent_representations_dir])

# father.npy와 mother.npy 파일의 존재 확인

father_image_name = "D:/SSLHK/testpoto/dad.jpg"  # 아버지 이미지 파일 이름
mother_image_name = "D:/SSLHK/testpoto/mom.jpg"  # 어머니 이미지 파일 이름

# father_latent_path = os.path.join(latent_representations_dir, f"father.npy")
father_latent_path = os.path.join(latent_representations_dir, f"dad_01.npy")
# mother_latent_path = os.path.join(latent_representations_dir, f"mother.npy")
mother_latent_path = os.path.join(latent_representations_dir, f"mom_01.npy")

# 생성된 latent representation 파일의 존재 확인
if os.path.exists(father_latent_path) and os.path.exists(mother_latent_path):
    print(f"Latent representations have been successfully created:/{father_latent_path}/{mother_latent_path}")
else: 
    print("Error: Unable to find the latent representation files. Please check the paths and try again.")
