import os
from tqdm import *
from skimage import io
from shutil import copyfile
import cv2
import numpy as np 
import imgaug as ia
from imgaug import augmenters as iaa
from skimage import transform as trans
from shutil import copyfile
import face_alignment
from PIL import Image
import io as io2

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

def alignment(cv_img, dst, dst_w, dst_h):
    src = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041] ], dtype=np.float32)
    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2,:]
    face_img = cv2.warpAffine(cv_img,M,(dst_w,dst_h), borderValue = 0.0)
    return face_img

def detect(image):
    image = io.imread(io2.BytesIO(image))
    landmarks = fa.get_landmarks(image)
    check = False

    if landmarks is None:
        for sigma in np.linspace(0.0, 3.0, num=11).tolist():
            seq = iaa.GaussianBlur(sigma)
            image_aug = seq.augment_image(image)
            landmarks = fa.get_landmarks(image_aug)
            if landmarks is not None:
                print('sigma:',sigma)
                points = landmarks[0]
                p1 = np.mean(points[36:42,:], axis=0)
                p2 = np.mean(points[42:48,:], axis=0)
                p3 = points[33,:]
                p4 = points[48,:]
                p5 = points[54,:]
                
                if np.mean([p1[1],p2[1]]) < p3[1] \
                    and p3[1] < np.mean([p4[1],p5[1]]) \
                    and np.min([p4[1], p5[1]]) > np.max([p1[1], p2[1]]) \
                    and np.min([p1[1], p2[1]]) < p3[1] \
                    and p3[1] < np.max([p4[1], p5[1]]):

                    dst = np.array([p1,p2,p3,p4,p5],dtype=np.float32)
                    cv_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    face = alignment(cv_img, dst, 112, 112)

                    check = True
                    break
    else:
        points = landmarks[0]
        p1 = np.mean(points[36:42,:], axis=0)
        p2 = np.mean(points[42:48,:], axis=0)
        p3 = points[33,:]
        p4 = points[48,:]
        p5 = points[54,:]

        dst = np.array([p1,p2,p3,p4,p5],dtype=np.float32)
        cv_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        face = alignment(cv_img, dst, 112, 112)

        check = True

    if check == True:
        return face
    else:
        return None






if __name__ == "__main__":
    image = io.imread('./img/218_copy.jpg')
    face = detect(image)
    if face is not None:
        cv2.imwrite('./test.png', face)