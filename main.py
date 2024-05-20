import numpy as np
from src.Utils.imgCrop import order_points, four_point_transform
from src.objectDetector import BoardDetector, BoardObject, TextArea
import cv2
import easyocr
from src.textRecognition import getPredictor, predictText

if __name__ == "__main__":
    boardDetector = BoardDetector()
    image = cv2.imread('/home/nqhuy/workBench/BTN-AI/img/sample04sample04.jpeg')
    image_draw_boudering, box = boardDetector.detect(image)
    mask = BoardDetector.genMask(image)
    objects = BoardDetector.getBoardObjects(image, box)

    reader = easyocr.Reader(['vi'], gpu=True)
    predictor = getPredictor()
    id = 0
    for i, obj in enumerate(objects):
        horizontal_list, free_list = reader.detect(obj.image)
        if (len(horizontal_list[0]) != 0):
            for box in horizontal_list[0]:
                id += 1
                x_min, x_max, y_min, y_max = box
                box = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
                box = np.array(box)
                cropped = four_point_transform(obj.image, order_points(box))
                obj.addTextArea(TextArea(cropped, box))
        if len(free_list[0]) != 0:
            for points in free_list[0]:
                id += 1
                points = order_points(np.array(points))
                cropped = four_point_transform(obj.image, points)
                obj.addTextArea(TextArea(cropped, points))
        for textArea in obj.textAreas:
            textArea.text = predictText(textArea.img, predictor)
        obj.genTextValue()
        print(f"object {i+1}: {obj.text}")
        cv2.putText(image_draw_boudering, f"object{i+1}", (obj.box[0][0], obj.box[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)

    cv2.imshow('Detected Signs', image_draw_boudering)

    cv2.waitKey(0)
    cv2.destroyAllWindows()