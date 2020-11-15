from typing import List
import cv2

class BoundingBox:
    left: int
    right: int
    top: int
    bottom: int
    
    def __init__(self, left, top, right, bottom):
        self.left = int(left)
        self.top = int(top)
        self.right = int(right)
        self.bottom = int(bottom)


class FaceDetection:
    bbox: BoundingBox
    category: int
    confidence: float

    def __init__(self, left, top, right, bottom, category, confidence):
        self.bbox = BoundingBox(left, top, right, bottom)
        self.category = int(category)
        self.confidence = confidence

    def __str__(self):
        return f'#FaceDetection# bbox=[({self.bbox.left},{self.bbox.top})->({self.bbox.right},{self.bbox.bottom})], confidence={self.confidence}'
    

class FaceDetector():

    def __init__(self):
        # TODO: add initialization logic
        pass

    def detect_image(self, img) -> List[FaceDetection]:
        # TODO: add detect logic for single image
        return [
            FaceDetection(180, 56, 244, 144, 0, 0.8),
            FaceDetection(660, 28, 720, 112, 0, 0.9)
        ]

    def detect_images(self, imgs) -> List[List[FaceDetection]]:
        # TODO: add detect logic for batch images
        return [
            [
                FaceDetection(0, 0, 50, 50, 0, 0.8),
                FaceDetection(100, 100, 200, 200, 0, 0.9)
            ],
            []
        ]
    
    def visualize(self, image, detection_list: List[FaceDetection], color=(0,0,255), thickness=5):
        img = image.copy()
        for detection in detection_list:
            bbox = detection.bbox
            p1 = bbox.left, bbox.top
            p2 = bbox.right, bbox.bottom
            cv2.rectangle(img, p1, p2, color, thickness=thickness, lineType=cv2.LINE_AA)
        return img
