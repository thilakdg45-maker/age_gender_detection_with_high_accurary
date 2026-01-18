import os
import cv2

def read_utkface_dataset(path):
    images = []
    ages = []
    genders = []

    for filename in os.listdir(path):
        if filename.endswith(".jpg"):
            parts = filename.split("_")
            age = int(parts[0])
            gender = int(parts[1])   # 0 = male, 1 = female

            img_path = os.path.join(path, filename)
            img = cv2.imread(img_path)

            if img is not None:
                images.append(img)
                ages.append(age)
                genders.append(gender)

    return images, ages, genders

dataset_path = r"C:\Users\Thilak D G\UTKFace\UTKFace-master\UTKFace"
images, ages, genders = read_utkface_dataset(dataset_path)

print("Total images loaded:", len(images))
