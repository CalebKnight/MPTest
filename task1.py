

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
import numpy as np
from image import Filter

def GetEdges(img):

    # For this purpose canny values are sensitive, given we want as little lines that are not part of the barcode or it's boundaries
    edges = cv2.Canny(img, 70, 280)
    

    # Flip the image, threshold then flip back
    edges = cv2.bitwise_not(edges)

    edges = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    edges = cv2.bitwise_not(edges)
    
    return edges 

def GetEdgesForText(img):
    # When we get edges for text its best to leave the edges in their 'raw' form
    edges = cv2.Canny(img, 50, 150)
    return edges

def toGrayscale(imgPath):
    # All images are resized to 1000x1000
    img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (1000,1000))
    return img


def LoadImage(imgPath):
    try:
        img = toGrayscale(imgPath)
        img = Filter(img)
    except Exception as e:
        print("Error loading image: ", imgPath)
        print(e)
        return None
    return img

def GetPointDistance(p1, p2):
    # get euclidean distance between two points
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def GetContours(img):
    # get contours 
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    approxContours = []
    for contour in contours:
        epsilon = 0.1 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        x, y, w, h = cv2.boundingRect(approx)
        # EAN 13 barcodes are dimensioned 25.93mm x 37.29mm
        # as such we should use this ratio (both ways w/h or h/w to determine if the barcode is valid)

        # We are generous but these values represent the typical barcode dimensions. As such we add 0.35 to the acceptable range
        if not (25.93 / 37.29) - 0.35 < w / h < (25.93 / 37.29) + 0.35:
            continue

        # We only want rectangles and they need to be big enough to be a barcode
        if len(approx) == 4  and cv2.contourArea(approx) > 55000:
            approxContours.append(approx)

    return approxContours

def GetTextContours(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # sort contours by size (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # max size should be width * 0.1
    maxW = 60
    maxH = 60
    minw = 20
    minH = 15
    left = []
    right = []
    for contour in contours:
        contour = cv2.boundingRect(contour)
        x, y, w, h = contour
        # aspect ratio of a barcode number should not be too wide or too tall
        aspect = w / h
        if aspect > 2.5 or aspect < 0.5:
            continue

        if minw < w < maxW and minH < h < maxH:
            if x < img.shape[1] // 2:
                left.append(contour)
            else:
                right.append(contour)
 


    return left, right



def ExpandPoint(pt, center, paddingX, paddingY):
    # Expand the point away from the center by the padding percentage
    return [int(pt[0] + (pt[0] - center[0]) * paddingX), int(pt[1] + (pt[1] - center[1]) * paddingY)]


# We dilate barcode lines to combine close lines into a single rectangle
def DilateBarcodeLines(rotated):
    # We only dilate in the y direction
    rotated = cv2.dilate(rotated, np.ones((3, 1), np.uint8), iterations=1)
    rotated = cv2.morphologyEx(rotated, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)   
    # We only erode in the x direction
    rotated = cv2.erode(rotated, np.ones((1, 3), np.uint8), iterations=1)
    return rotated

def SortContourPoints(contour):
    YSorted = sorted(contour, key=lambda x: x[0][1])
    XSorted = sorted(YSorted[:2], key=lambda x: x[0][0])
    topLPoint = XSorted[0][0]
    topRPoint = XSorted[1][0]
    xSorted = sorted(YSorted[2:], key=lambda x: x[0][0])
    botLPoint = xSorted[0][0]
    botRPoint = xSorted[1][0]
    return topLPoint, topRPoint, botLPoint, botRPoint


def PerformTransformWithRectPoints(rect, img, expand=0):
    topLPoint, topRPoint, botLPoint, botRPoint = SortContourPoints(rect)
   
    # We need to get the distance of points for later
    topWidth = GetPointDistance(topLPoint, topRPoint)
    botWidth = GetPointDistance(botLPoint, botRPoint)
    leftHeight = GetPointDistance(topLPoint, botLPoint)
    rightHeight = GetPointDistance(topRPoint, botRPoint)

    # Padding is required, because we miss the text when we extract the barcode on this pass. 
    if expand:
        # Find the center of the detected rectangle
        centerX = (topLPoint[0] + botRPoint[0]) // 2
        centerY = (topLPoint[1] + botRPoint[1]) // 2
        centerPoint = [centerX, centerY]
        # Expand each point by the padding factor
        topLPoint = ExpandPoint(topLPoint, centerPoint, expand, expand)
        topRPoint = ExpandPoint(topRPoint, centerPoint, expand, expand)
        botLPoint = ExpandPoint(botLPoint, centerPoint, expand, expand)
        botRPoint = ExpandPoint(botRPoint, centerPoint, expand, expand)

    

    # Max width and height for transformed image
    maxWidth = max(int(topWidth), int(botWidth))
    maxHeight = max(int(leftHeight), int(rightHeight))

    # We need to get the destination points for the perspective transform
    dst_points = np.float32([[0, 0], [maxWidth - 1, 0], [0, maxHeight - 1], [maxWidth - 1, maxHeight - 1]])

    # We need to get the source points for the perspective transform
    src_points = np.float32([topLPoint, topRPoint, botLPoint, botRPoint])
    
    # Perform the transform, thereby extracting just the barcode and it's immediate neighbouring text (some noise will be extracted, given this is an estimation)
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    transformed = cv2.warpPerspective(img, matrix, (maxWidth, maxHeight))
    return transformed, src_points, dst_points


def GetBarcode(path):
    print("\n\n")
    if not path.endswith('.png') and not path.endswith('.jpg'):
        return None, None, None, None, None
    print("Processing: ", path)

    original = cv2.imread(path)
    processed = LoadImage(path)
    if processed is None:
        return None, None, None, None, None

    finalRect = [[0, 0], [0, 0], [0, 0], [0, 0]]

    # We get edges in the image
    edges = GetEdges(processed)

    # We look for lines that are reasonably large because these will most likely be the barcode lines. Given the barcode lines dominate an image this allows us to find the angle of the barcode
    hough = cv2.HoughLinesP(edges, 1, np.pi/720, 50, minLineLength=75, maxLineGap=5)


    if hough is None:
        return None, None, None, None, None

    angles = {}
    for line in hough:
        x1, y1, x2, y2 = line[0]
        # This line was generated by ChatGPT 4.0 to do the math for the angle of a line
        angle = math.degrees(math.atan2(y2-y1, x2-x1))
        angles[angle] = angles.get(angle, []) + [line]


    # We find the most common angle so we can rotate the image to match the barcodes 'orientation' 
    # IE we rotate it so that the barcode is always horizontal
    most = max(angles, key=lambda x: len(angles[x]))
    rotationMatrix = cv2.getRotationMatrix2D((processed.shape[1] / 2, processed.shape[0] / 2), most, 1)
    rotated = cv2.warpAffine(edges, rotationMatrix, (edges.shape[1], edges.shape[0]))


    # By repeatidly dilating barcode lines until we find a contour that is like a barcode, we can find the barcode
    # This method is flawed, given its very sensitive to the original edge map. (Missing edges cause BIG problems)
    i = 0
    found = True
    print("Finding barcode by dilating lines: ", path)
    while GetContours(rotated) == []:
        i += 1
        rotated = DilateBarcodeLines(rotated)
        if i > 25:
            found = False
            break
    if not found:
        return None, None, None, None, None

    # We get our rectangular contours here and then sort them, largest first (most likely that largest rectangular contour is the barcode)
    contours = GetContours(rotated)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # We rotate the original image to match the barcode
    processed = cv2.warpAffine(processed, rotationMatrix, (processed.shape[1], processed.shape[0]))


    transformed = None
    dst_points = None
    src_points = None
    print("Extracting the barcode area: ", path)
    for contour in contours:
        # The entirety of this sorting is to ensure the rectangle's points are in the correct order
        transformed, src_points, dst_points = PerformTransformWithRectPoints(contour, processed, expand=0.2)


    if transformed is None:
        return None, None, None, None, None
    

    print("Extracting text area: ", path)
    transformed = GetEdgesForText(transformed)
    originalH, originalW = transformed.shape
    transformed = cv2.resize(transformed, (500, 500))
    # Sharpen the image to make the text more clear
    sharpeningKernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    transformed = cv2.filter2D(transformed, -1, sharpeningKernel)
    left,right = GetTextContours(transformed)

    # We take the longest list of contours as the text
    # The longest list will most likely be the barcode number side unless noisy text has made its way through
    text = max(left, right, key=lambda x: len(x))
    transformed = cv2.cvtColor(transformed, cv2.COLOR_GRAY2BGR)
    text = sorted(text, key=lambda x: x[1] + x[3])

    # We will never detect all text so we just make sure there is atleast 5 contours
    if len(text) < 5:
        return None, None, None, None, None

    # The first text and last text are the top and bottom numbers, this is sensitive to noise
    x1,y1,w1,h1 = text[0]
    x2,y2,w2,h2 = text[-1]   

    # We need our co-ordinates so we can reverse the perspective transform
    finalRect = [[x1 + w1, y1], [x1, y1], [x2, y2 + h2],  [x2 + w2, y2 + h2]]
    finalRect = np.array(finalRect, dtype='int32')

    # Reverse resizing when identifying the text
    for point in finalRect:
        point[0] = int(point[0] * originalW / 500)
        point[1] = int(point[1] * originalH  / 500)

    # Perform all the transforms in reverse to get the final rectangle
    finalRect = np.array(finalRect, dtype='float32')
    matrix = cv2.getPerspectiveTransform(dst_points, src_points)
    finalRect = cv2.perspectiveTransform(finalRect.reshape(-1, 1, 2), matrix)
    rotationMatrix = cv2.getRotationMatrix2D((processed.shape[1] / 2, processed.shape[0] / 2), -most, 1)
    finalRect = cv2.transform(finalRect, rotationMatrix)

    topLPoint, topRPoint, botLPoint, botRPoint = SortContourPoints(finalRect)

    # Finally we need to count for our first resize to 1000x1000
    for point in [topLPoint, topRPoint, botLPoint, botRPoint]:
        point[0] = int(point[0] * original.shape[1] / 1000)
        point[1] = int(point[1] * original.shape[0] / 1000)

    

    warped = PerformTransformWithRectPoints(finalRect, original)[0]

    if warped.shape[0] > warped.shape[1]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
        print("Rotating extracted barcode image")

    return topLPoint, topRPoint, botLPoint, botRPoint, warped

def run_task1(image_path, config):
    count = 0
    negCount = 0
    for imgIdx, path in enumerate(os.listdir(image_path)):

        if not path.endswith('.png') and not path.endswith('.jpg'):
            continue

        
        topLPoint, topRPoint, botLPoint, botRPoint, warped = GetBarcode(image_path + "/" + path)

        if topLPoint is None:
            print(f"No barcode found in {path}")
            negCount += 1
            continue

        print(f"Barcode found in {path}")
        count += 1
        print(f'Writing to output/task1/img{imgIdx + 1}.txt')
        # cords = f'{int(topLPoint[0])},{int(topLPoint[1])},{int(topRPoint[0])},{int(topRPoint[1])},{int(botLPoint[0])},{int(botLPoint[1])},{int(botRPoint[0])},{int(botRPoint[1])}'
        # This is the format used in the demonstration
        cords = f'{int(topLPoint[0])},{int(topLPoint[1])},{int(botLPoint[0])},{int(botLPoint[1])},{int(botRPoint[0])},{int(botRPoint[1])},{int(topRPoint[0])},{int(topRPoint[1])}'
        save_output(f'output/task1/img{imgIdx + 1}.txt', cords, 'txt')
        print(f'Writing to output/task1/img{imgIdx + 1}.png')
        save_output(f'output/task1/img{imgIdx + 1}.png', warped, 'image')
    

def save_output(output_path, content, output_type='txt'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if output_type == 'txt':
        with open(output_path, 'w') as f:
            f.write(content)
        print(f"Text file saved at: {output_path}")
    elif output_type == 'image':
        # Assuming 'content' is a valid image object, e.g., from OpenCV
        cv2.imwrite(output_path, content)
        print(f"Image saved at: {output_path}")
    else:
        print("Unsupported output type. Use 'txt' or 'image'.")
