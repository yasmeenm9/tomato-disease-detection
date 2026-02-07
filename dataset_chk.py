import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

DATASET_PATH = "dataset/train"   # change if your path name differs

img_height = 224
img_width = 224
batch_size = 32

train_ds = image_dataset_from_directory(
    DATASET_PATH,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names
print("Classes:", class_names)
print("Number of classes:", len(class_names))
