
import os
import bz2
from keras.utils import get_file
from ffhq_dataset.face_alignment import image_align
from ffhq_dataset.landmarks_detector import LandmarksDetector
import multiprocessing

def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path

def align_images(raw_dir, aligned_dir, output_size=1024, x_scale=1, y_scale=1, em_scale=0.1, use_alpha=False):
    landmarks_model_path = unpack_bz2("shape_predictor_68_face_landmarks.dat.bz2")
    landmarks_detector = LandmarksDetector(landmarks_model_path)
    for img_name in os.listdir(raw_dir):
        print('Aligning %s ...' % img_name)
        try:
            raw_img_path = os.path.join(raw_dir, img_name)
            fn = face_img_name = '%s_%02d.png' % (os.path.splitext(img_name)[0], 1)
            face_img_path = os.path.join(aligned_dir, face_img_name)
            
            # For face alignment
            image_align(raw_img_path, face_img_path, landmarks_model_path, output_size, x_scale, y_scale, em_scale, use_alpha)
        except Exception as e:
            print('Exception:', e)
            continue
