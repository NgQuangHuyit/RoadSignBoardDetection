import cv2
import numpy as np
from src.Utils.imgCrop import order_points, four_point_transform
from typing import List

class TextArea:
    def __init__(self, img: np.ndarray, box: np.ndarray, text=None):
        self._img = img
        self._box = box
        self._text = text

    @property
    def img(self):
        return self._img

    @property
    def box(self):
        return self._box

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        self._text = value


class BoardObject:
    def __init__(self, image, box):
        # Ảnh của đối tượng
        # Tọa dộ 4 diểm của đối tượng
        self.image = image
        self.box = box
        self.textAreas = []
        self.text = None

    
    def addTextArea(self, textArea: TextArea):
        self.textAreas.append(textArea)

    def getTextAreas(self) -> List[TextArea]:
        return self.textAreas
    
    def getImage(self):
        return self.image

    def genTextValue(self):
        self.textAreas.sort(key=lambda x: x.box[0][1])
        text = ''
        for textArea in self.textAreas:
            text += textArea.text + ' '
        self.text = text
        return text


class BoardDetector:
    def __init__(self, approx_cnt=4, min_area=2000, aspect_ratio=(0.8, 3), black_pix_ratio=(0.2, 0.85)):
        """
            approx_cnt: Số cạnh xấp xỉ của đối tượng
            min_area: Diện tích tối thiểu của đối tượng
            aspect_ratio: Tỷ lệ khung hình của đối tượng
            black_pix_ratio: Tỷ lệ pixel màu đen trong đối tượng
        """
        self.approx_cnt = approx_cnt
        self.min_area = min_area
        self.aspect_ratio = aspect_ratio
        self.black_pix_ratio = black_pix_ratio

    @staticmethod
    def denoise(binary_image):
    # Sử dụng phép toán morphology để loại bỏ nhiễu
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        denoised_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
        return denoised_image 
    
    @staticmethod
    def grayscale_and_binary(image, threshold=127):
        # Đọc ảnh
        # Chuyển đổi thành ảnh xám
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Nhị phân hóa ảnh
        _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
        
        return binary_image
    
    @staticmethod
    def genMask(image, lower=np.array([90, 50, 50]), upper=np.array([130, 255, 255])):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        kernel = np.ones((5,5),np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = BoardDetector.denoise(mask)
        return mask
    
    def detect(self, image, mask=None):
        
        image = image.copy()
        if mask is None:
            mask = BoardDetector.genMask(image)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        box_list = []
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            epsilon = 0.05 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Kiểm tra xem contour có 4 cạnh không
            if len(approx) != 4:
                continue
            number_of_black_pix = np.sum(mask[y:y+h, x:x+w]) / 255 
            if (number_of_black_pix/(w*h)) > self.black_pix_ratio[1] or (number_of_black_pix/(w*h)) < self.black_pix_ratio[0]:
                continue
            if (w/h > self.aspect_ratio[1]) or (w/h < self.aspect_ratio[0]):
                continue
            
            if (w*h) < self.min_area:
                continue
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            box_list.append(box)
            cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
        return image, box_list
    
    @staticmethod
    def crop(image, box):
        return four_point_transform(image, order_points(box))

    @staticmethod
    def getBoardObjects(image, box_list) -> List[BoardObject]:
        board_objects = []
        for box in box_list:
            board_objects.append(BoardObject(BoardDetector.crop(image, box), box))
        return board_objects

if __name__ == "__main__":
    detector = BoardDetector()
    image = cv2.imread('/home/nqhuy/workBench/BTN-AI/img/test5test5.jpg')
    image_draw, box_list = detector.detect(image)
    objects = BoardDetector.getBoardObjects(image, box_list)
    for i, obj in enumerate(objects):
        cv2.imshow(f'Object_{i}', obj.image)
    cv2.imshow('result', image_draw)
    cv2.imshow("original", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()