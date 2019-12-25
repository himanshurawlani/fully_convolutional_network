import os
import numpy as np
import cv2
from sklearn import preprocessing
import tensorflow as tf


class Generator(tf.keras.utils.Sequence):

    def __init__(self, DATASET_PATH, BATCH_SIZE=32, shuffle_images=True, image_min_side=24):
        """ Initialize Generator object.

        Args
            DATASET_PATH           : Path to folder containing individual folders named by their class names
            BATCH_SIZE             : The size of the batches to generate.
            shuffle_images         : If True, shuffles the images read from the DATASET_PATH
            image_min_side         : After resizing the minimum side of an image is equal to image_min_side.
        """

        self.batch_size = BATCH_SIZE
        self.shuffle_images = shuffle_images
        self.image_min_side = image_min_side
        self.load_image_paths_labels(DATASET_PATH)
        self.create_image_groups()
    
    def load_image_paths_labels(self, DATASET_PATH):
        
        classes = os.listdir(DATASET_PATH)
        lb = preprocessing.LabelBinarizer()
        lb.fit(classes)

        self.image_paths = []
        self.image_labels = []
        for class_name in classes:
            class_path = os.path.join(DATASET_PATH, class_name)
            for image_file_name in os.listdir(class_path):
                self.image_paths.append(os.path.join(class_path, image_file_name))
                self.image_labels.append(class_name)

        self.image_labels = np.array(lb.transform(self.image_labels), dtype='float32')
        
        assert len(self.image_paths) == len(self.image_labels)

    def create_image_groups(self):
        if self.shuffle_images:
            # Randomly shuffle dataset
            seed = 4321
            np.random.seed(seed)
            np.random.shuffle(self.image_paths)
            np.random.seed(seed)
            np.random.shuffle(self.image_labels)

        # Divide image_paths and image_labels into groups of BATCH_SIZE
        self.image_groups = [[self.image_paths[x % len(self.image_paths)] for x in range(i, i + self.batch_size)]
                              for i in range(0, len(self.image_paths), self.batch_size)]
        self.label_groups = [[self.image_labels[x % len(self.image_labels)] for x in range(i, i + self.batch_size)]
                              for i in range(0, len(self.image_labels), self.batch_size)]

    def resize_image(self, img, min_side_len):

        h, w, c = img.shape

        # limit the min side maintaining the aspect ratio
        if min(h, w) < min_side_len:
            im_scale = float(min_side_len) / h if h < w else float(min_side_len) / w
        else:
            im_scale = 1.

        new_h = int(h * im_scale)
        new_w = int(w * im_scale)

        re_im = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return re_im, new_h / h, new_w / w

    def load_images(self, image_group):

        images = []
        for image_path in image_group:
            img = cv2.imread(image_path)
            img_shape = len(img.shape)
            if img_shape == 2:
                img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
            elif img_shape == 4:
                img = cv2.cvtColor(img,cv2.COLOR_BGRA2RGB)
            elif img_shape == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img, rh, rw = self.resize_image(img, self.image_min_side)
            images.append(img)

        return images

    def construct_image_batch(self, image_group):
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))

        # construct an image batch object
        image_batch = np.zeros((self.batch_size,) + max_shape, dtype='float32')

        # copy all images to the upper left part of the image batch object
        for image_index, image in enumerate(image_group):
            image_batch[image_index, :image.shape[0], :image.shape[1], :image.shape[2]] = image

        return image_batch
    
    def __len__(self):
        """
        Number of batches for generator.
        """

        return len(self.image_groups)

    def __getitem__(self, index):
        """
        Keras sequence method for generating batches.
        """
        image_group = self.image_groups[index]
        label_group = self.label_groups[index]
        images = self.load_images(image_group)
        image_batch = self.construct_image_batch(images)

        return np.array(image_batch), np.array(label_group)

if __name__ == "__main__":

    BASE_PATH = 'dataset'
    train_generator = Generator('dataset/train')
    val_generator = Generator('dataset/val')
    print(len(train_generator))
    print(len(val_generator))
    image_batch, label_group = train_generator.__getitem__(0)
    print(image_batch.shape)
    print(label_group.shape)
    