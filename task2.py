

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

import math
import os

import cv2


from image import GetEdgesForText

def GetPointDistance(p1, p2):
    # get euclidean distance between two points
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def GetContours(img):
    # Find contours
    # We need to dilate the image so that our contours can detect 3's e.t.c
    img = cv2.dilate(img, None, iterations=1)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def toGrayscale(imgPath):
    img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
    return img


def LoadImage(imgPath):
    img = toGrayscale(imgPath)
    return img

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

def GetNumbers(img, path=""):
    
    edges = GetEdgesForText(img)
    cv2.imshow("Edges", edges)
    contours = GetContours(edges)
    numbers = []

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # sort left to right
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])
    rectangles = []
    for idx,contour in enumerate(contours):
        x1, y1, w, h = cv2.boundingRect(contour)
        # Text is usually quite tall and not too wide or too thin
        if  h >= img.shape[0] * 0.7  and w > img.shape[1] *0.01 and w < img.shape[1] * 0.15:
            numbers.append((x1, y1, x1 + w, y1 + h))

    # sort numbers by x1 so we get contours in order
    numbers = sorted(numbers, key=lambda x: x[0])
    # Restrict to 13 numbers given the barcode cannot have more than 13 numbers
    numbers = numbers[:13]
    return numbers

def run_task2(image_path, config):
    
    for path in os.listdir(image_path):
        if not path.endswith('.png') and not path.endswith('.jpg'):
            continue

        img = LoadImage(f'data/validation/test/task2/{path}')
        original = img.copy()
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        numbers = GetNumbers(img, path)
        for idx,number in enumerate(numbers):
            x1, y1, x2, y2 = number
            number_img = img[y1:y2, x1:x2]
            outputBar = path.split(".png")[0]
            if idx < 9:
                idx = f"0{idx + 1}"
            else:
                idx = str(idx + 1)
            if not os.path.exists(f"output/task2/{outputBar}"):
                os.makedirs(f"output/task2/{outputBar}")
            outputPath = f"output/task2/{outputBar}/d{idx}.png"
            cv2.imwrite(outputPath, number_img)
            print(f"Saved number {number} to {outputPath}")
            outputPath = f"output/task2/{outputBar}/d{idx}.txt"
            with open(outputPath, "w") as f:
                nums = [str(num) for num in number]
                f.write(",".join(nums))
                print(f"Saved number {number} to {outputPath}")
        print(f"Number of numbers in {path}: {len(numbers)}")
