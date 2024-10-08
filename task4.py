

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
import numpy as np

from task1 import GetBarcode
from task2 import GetNumbers
from task3 import LoadModel
from image import Filter, GetEdgesForText


def toGrayscale(img):
    img = cv2.resize(img, (100, 100))
    return img


def LoadImage(imgPath):
    img = toGrayscale(imgPath)
    img = Filter(img)
    return img



def run_task4(image_path, config):
    model = LoadModel()
    for imgIdx, path in enumerate(os.listdir(image_path)):
        # Task 1
        _, _, _, _, warped = GetBarcode(f"{image_path}/{path}")
        if warped is None:
            print(f"Barcode not found in image {imgIdx + 1}")
            continue

        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
        # Task 2
        numbers = GetNumbers(warped)
        print(len(numbers))
        number_images = []
        for number in numbers:
            x1, y1, x2, y2 = number
            number_img = warped[y1:y2, x1:x2]
            number_images.append(number_img)

        predictions = []

        # Task 3
        for img in number_images:
            img = LoadImage(img)
            edges = GetEdgesForText(img)
            edges = np.array([edges])
            prediction = model.predict(edges)
            predictions.append(np.argmax(prediction))

        if len(predictions) < 1:
            print(f"No numbers found in image {imgIdx + 1}")
            continue


        if len(predictions) < 13 and predictions[0] != 9:
            predictions.insert(0, 9)

        if predictions[0] != 9:
            predictions[0] = 9

        # Output
        print(f"Predictions for image {imgIdx + 1}: {predictions}")
        strPredictions = str(predictions).strip('[]').replace(',', '').replace(" ", "")
        output_path = f"output/task4/img{imgIdx + 1}.txt"
        save_output(output_path, strPredictions, output_type='txt')


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
