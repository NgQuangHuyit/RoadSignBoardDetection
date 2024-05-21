import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from src.Utils.imgCrop import order_points, four_point_transform
from src.objectDetector import BoardDetector, TextArea
import cv2
import easyocr
from src.textRecognition import getPredictor, predictText


def processImage(path):
    result = ""
    boardDetector = BoardDetector()
    image = cv2.imread(path)
    result_img, box = boardDetector.detect(image)
    objects = BoardDetector.getBoardObjects(image, box)
    # Khởi tạo easyocr reader
    reader = easyocr.Reader(['vi'], gpu=True)
    predictor = getPredictor()

    for i, obj in enumerate(objects):
        horizontal_list, free_list = reader.detect(obj.image)
        if len(horizontal_list[0]) != 0:
            for box in horizontal_list[0]:
                x_min, x_max, y_min, y_max = box
                box = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
                box = np.array(box)
                cropped = four_point_transform(obj.image, order_points(box))
                obj.addTextArea(TextArea(cropped, box))
        if len(free_list[0]) != 0:
            for points in free_list[0]:
                points = order_points(np.array(points))
                cropped = four_point_transform(obj.image, points)
                obj.addTextArea(TextArea(cropped, points))
        for textArea in obj.textAreas:
            textArea.text = predictText(textArea.img, predictor)
        obj.genTextValue()
        result += f"object {i + 1}: {obj.text}\n"
        cv2.putText(result_img, 
                    f"object{i + 1}", 
                    (obj.box[0][0], obj.box[0][1]), 
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0,255), 2)
        
    result_img = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    original_img = Image.open(path)
    original_img = original_img.resize((400, 400))
    original_photo = ImageTk.PhotoImage(original_img)
    original_label.config(image=original_photo)
    original_label.image = original_photo
    result_img = result_img.resize((400, 400))
    result_photo = ImageTk.PhotoImage(result_img)
    result_label.config(image=result_photo)
    result_label.image = result_photo
    description_label.config(text=result)


def imageUploader():
    fileTypes = [("JPEG files", "*.jpg"), ("JPEG files", "*.jpeg"), ("PNG files", "*.png")]
    path = tk.filedialog.askopenfilename(filetypes=fileTypes)
    if len(path):
        processImage(path)


if __name__ == "__main__":
    app = tk.Tk()
    app.title("Image Processor")
    app.geometry("600x350")
    original_label = tk.Label(app)
    original_label.pack(side=tk.TOP, padx=10, pady=10)
    result_label = tk.Label(app)
    result_label.pack(side=tk.TOP, padx=10, pady=10)
    description_label = tk.Label(app, text="")
    description_label.pack(side=tk.TOP, padx=10, pady=10)
    uploadButton = tk.Button(app, text="Upload Image", command=imageUploader)
    uploadButton.pack(pady=20)
    app.mainloop()