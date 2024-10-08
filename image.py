import cv2
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

def Filter(img):
    # normalise brightness and contrast for the image
    img = cv2.equalizeHist(img)
    # Some images may be blurry so blurring the image will remove noise
    img = cv2.GaussianBlur(img, (5, 5), 0)
    # Some images may have salt and pepper noise so median blur will remove this
    img = cv2.medianBlur(img, 5)
    # normalise the image
    img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return img

def GetEdgesForText(img):
    
        edges = cv2.Canny(img, 30, 90)
        
        edges = cv2.bitwise_not(edges)
    
        edges = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
        edges = cv2.bitwise_not(edges)
        
        return edges