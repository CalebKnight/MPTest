

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Author: Caleb Knight
# Last Modified: 2024-08-10

import os

import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from image import Filter, GetEdgesForText



def toGrayscale(imgPath):
    img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (100, 100))
    return img

def LoadImage(imgPath):
    img = toGrayscale(imgPath)
    img = Filter(img)
    return img


# For images that are mostly black, we don't want to use them for training, given the model won't learn anything
def IsMostlyBlack(img):
    black = 0
    for row in img:
        for pixel in row:
            if pixel == 0:
                black += 1
    return black > 0.8 * img.size


def MakeMoreData(images, labels):
    new_images = []
    new_labels = []

    # Dilation 
    for i in range(len(images)):
        dilated_image = cv2.dilate(images[i], np.ones((3, 3), np.uint8), iterations=1)
        new_images.append(dilated_image)
        new_labels.append(labels[i])

    # Brightness adjustment
    for i in range(len(images)):
        for factor in [1.2, 1.5, 1.8]:
            bright_image = cv2.convertScaleAbs(images[i], alpha=factor, beta=0)
            if not IsMostlyBlack(bright_image):
                new_images.append(bright_image)
                new_labels.append(labels[i])

    # Contrast adjustment
    for i in range(len(images)):
        for offset in [30, 50, 70]:
            contrast_image = cv2.convertScaleAbs(images[i], alpha=1.0, beta=offset)
            if not IsMostlyBlack(contrast_image):
                new_images.append(contrast_image)
                new_labels.append(labels[i])

    # Erosion 
    for i in range(len(images)):
        eroded_image = cv2.erode(images[i], np.ones((3, 3), np.uint8), iterations=1)
        if not IsMostlyBlack(eroded_image):
            new_images.append(eroded_image)
            new_labels.append(labels[i])

    # Laplacian edge detection (for adding noise)
    for i in range(len(images)):
        laplacian_image = cv2.Laplacian(images[i], cv2.CV_64F)
        laplacian_image = cv2.convertScaleAbs(laplacian_image)  
        if not IsMostlyBlack(laplacian_image):
            new_images.append(laplacian_image)
            new_labels.append(labels[i])
        
    return new_images, new_labels

def RandomBarcode(idx):
    # The below code for swapping was produced by GPT 4.0
    nums = [0,1,2,3,4,5,6,7,8,9,idx,idx,idx,idx]
    for i in range(20):
        idx1 = np.random.randint(0, 9)
        idx2 = np.random.randint(0, 9)
        temp = nums[idx1]
        nums[idx1] = nums[idx2]
        nums[idx2] = temp
    print(nums)
    nums[0] = 9
    return nums

def GetXY():
    xy = {}

    for barcode in range(1,5):
        xy[barcode] = {"images": [], "labels": []}
        for path in os.listdir(f'data/validation/test/task3/barcode{barcode}'):
            if path.endswith('.txt'):
                file = open(f'data/validation/test/task3/barcode{barcode}/{path}', 'r')
                labels = file.read().split(",")
                labels = [int(label) for label in labels]
                xy[barcode]["labels"] = labels
                file.close()
                continue
            img = LoadImage(f'data/validation/test/task3/barcode{barcode}/{path}')
            edges = GetEdgesForText(img)
            xy[barcode]["images"].append(edges)
    return xy

def TrainModel():
    xy = GetXY()
    newXY = {}

    # We want to make bins for each digit and it's associated images for later use
    print("Creating bins for digits")
    bins = {}
    for i in range(0,10):
        bins[i] = {"images": [], "labels": []}
    for barcode in range(1, 5):
        for i in range(len(xy[barcode]["images"])):
            bins[xy[barcode]["labels"][i]]["images"].append(xy[barcode]["images"][i])
            bins[xy[barcode]["labels"][i]]["labels"].append(xy[barcode]["labels"][i])
    
    print("Creating new barcode values")
    for i in range(1, 5):
        newXY[i] = {"images": [], "labels": []}
        for val in RandomBarcode(i):
            newXY[i]["images"].extend(bins[val]["images"])
            newXY[i]["labels"].extend(bins[val]["labels"])


    # Need layers for dropout, batch normalization, and regularization to prevent overfitting
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.6),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # remove original xy values from xy, given we don't want to make our model too biased towards the original data
    for idx in newXY:
        newXY[idx]["images"], newXY[idx]["labels"] = MakeMoreData(newXY[idx]["images"], newXY[idx]["labels"])

    X = np.concatenate([np.array(newXY[key]["images"]) for key in newXY], axis=0)
    y = np.concatenate([np.array(newXY[key]["labels"]) for key in newXY], axis=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    loss = tf.keras.losses.SparseCategoricalCrossentropy(
    reduction="sum",
    name="sparse_categorical_crossentropy",
)
    # learning rate dynamically adjusts 
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=10000,
        decay_rate=0.9)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # Stop early to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    print("Compiling, training, and testing model")
    model.compile(optimizer, loss=loss, metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=50, batch_size=3, validation_split=0.2, callbacks=[early_stopping])

    test_loss, test_acc = model.evaluate(X_test, y_test)

    print('Test accuracy:', test_acc)
        
    print("Saving model")
    SaveModel(model)

    return model

def SaveModel(model):
    model.save('data/validation/model/model.h5')

def LoadModel():
    return tf.keras.models.load_model('data/validation/model/model.h5')

def run_task3(image_path, config):
    model = LoadModel()
    for path in os.listdir(image_path):
        predictions = []
        for imgIdx, path2 in enumerate(os.listdir(f'{image_path}/{path}')):

            # get labels from text file
            if path2.endswith('.txt'):
                file = open(f'{image_path}/{path}/labels.txt', 'r')
                labels = file.read().split(",")
                labels = [int(label) for label in labels]
                file.close()
                for prediction in predictions:
                    if prediction == labels[predictions.index(prediction)]:
                        print("Correct")
                    else:
                        print("Incorrect")
                        print(f"Prediction:{prediction}")
                        print(f"Actual label: {labels[predictions.index(prediction)]}")
                continue

            if not path2.endswith('.png') and not path2.endswith('.jpg'):
                continue
            print(f"Predicting for {path2}")
            
            img = LoadImage(f'{image_path}/{path}/{path2}')
            edges = GetEdgesForText(img)
            edges = np.array([edges])
            prediction = model.predict(edges)
            predictions.append(np.argmax(prediction))
            
            idx = imgIdx
            if idx < 9:
                idx = f"0{idx + 1}"
            else:
                idx = str(idx + 1)

            if not os.path.exists(f'output/task3/{path}'):
                os.makedirs(f'output/task3/{path}')
            print(f"Saving prediction to output/task3/{path}/{idx}.txt")
            save_output(f'output/task3/{path}/{idx}.txt', str(np.argmax(prediction)), 'txt')


def save_output(output_path, content, output_type='txt'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if output_type == 'txt':
        with open(output_path, 'w') as f:
            f.write(content)
        print(f"Text file saved at: {output_path}")
    elif output_type == 'image':
        # Assuming 'content' is a valid image object, e.g., from OpenCV
        content.save(output_path)
        print(f"Image saved at: {output_path}") 
    else:
        print("Unsupported output type. Use 'txt' or 'image'.")
           