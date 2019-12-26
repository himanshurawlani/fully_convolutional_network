import requests
import json
import os
import cv2
import sys
import numpy as np

classes = {0: 'daisy', 1: 'dandelion', 2: 'rose', 3: 'sunflower', 4: 'tulip'}

def decode_predictions(predictions):
    labels = []
    for preds in predictions:
        labels.append(classes[np.argmax(preds)])

    return labels

def make_serving_request(image_batch):
    data = json.dumps({"signature_name": "serving_default",
                       "instances": image_batch.tolist()})

    headers = {"content-type": "application/json"}

    os.environ['NO_PROXY'] = 'localhost'
    json_response = requests.post(
        'http://localhost:8501/v1/models/flower_classifier:predict', data=data, headers=headers)

    predictions = json.loads(json_response.text)['predictions']

    return predictions

def construct_image_batch(image_group, BATCH_SIZE):
    # get the max image shape
    max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))

    # construct an image batch object
    image_batch = np.zeros((BATCH_SIZE,) + max_shape, dtype='float32')

    # copy all images to the upper left part of the image batch object
    for image_index, image in enumerate(image_group):
        image_batch[image_index, :image.shape[0], :image.shape[1], :image.shape[2]] = image

    return image_batch

def find_type(image):
    
    image_batch = construct_image_batch(image_group, len(image_group))
    predictions = make_serving_request(image_batch)
    labels = decode_predictions(predictions)

    return labels
    

if __name__=="__main__":
    """
    Docker command to start tensorflow serving server:
    (NOTE: Run the below command inside "./flower_classifier" directory)
    $ docker run --rm -t -p 8501:8501 -v "$(pwd):/models/flower_classifier" -e MODEL_NAME=flower_classifier --name flower_classifier tensorflow/serving
    """

    image_group = []
    image_group.append(cv2.imread("test_images/sunflower.jpg")[:,:,::-1])
    image_group.append(cv2.imread("test_images/dandelion.jpg")[:,:,::-1])
    predictions = find_type(image_group)
    print(predictions)
