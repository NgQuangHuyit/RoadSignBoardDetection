import cv2
import easyocr
import numpy as np
from testVietOcr import getPredictor, predictText

class Sentence:
    def __init__(self, img: np.ndarray, lt, rt, rb, lb, text=None):
        self.img = img
        self.lt = lt
        self.rt = rt
        self.rb = rb
        self.lb = lb
        self.text = text

    def __str__(self):
        return f'{self.text} - ({self.x_min}, {self.y_min}) - ({self.x_max}, {self.y_max})'

    def __repr__(self):
        return f'{self.text} - ({self.x_min}, {self.y_min}) - ({self.x_max}, {self.y_max})'


def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped

image = cv2.imread('img/crop_img2.jpg')


reader = easyocr.Reader(['vi'], gpu=True)

horizontal_list, free_list = reader.detect(image)
# imgList = []
# Vẽ các hộp giới hạn trên ảnh
print(horizontal_list)
print(free_list)
id = 0
listWord = []
if (len(horizontal_list[0])!=0):
    for box in horizontal_list[0]:
        id += 1
        x_min, x_max, y_min, y_max = box  
        cutimg = image[y_min:y_max, x_min:x_max]
        cv2.imshow(f'word_{id}.jpg', cutimg)
        cv2.imwrite(f'./img/word_{id}.jpg', cutimg)
        listWord.append(Sentence(cutimg, (x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)))

if len(free_list[0]) != 0:
    for points in free_list[0]:
        id+=1
        points = order_points(np.array(points))
        print(points)
        warped = four_point_transform(image, points)
        # cv2.imshow('wraped', warped)
        cv2.imwrite(f'./img/wraped{id}.jpg', warped)
        listWord.append(Sentence(warped, points[0], points[1], points[2], points[3]))

pedictor = getPredictor()

for i, sentence in enumerate(listWord):
    cv2.imshow(f'word_{i}.jpg', sentence.img)
    text = predictText(sentence.img, pedictor)

    sentence.text = text
    cv2.imshow(f'word_{i}.jpg', sentence.img)
    print(f'Text: {text}')



# from vietocr.tool.predictor import Predictor
# from vietocr.tool.config import Cfg
# from PIL import Image
# import matplotlib.pyplot as plt

# # sử dụng config của các bạn được export lúc train nếu đã thay đổi 
# config = Cfg.load_config_from_name('vgg_transformer')
# # tham số
# # config = Cfg.load_config_from_file('config.yml')
# # sử dụng config mặc định của mô hình
# # config['weights'] = '/home/nqhuy/IdeaProjects/BT-AI/weights/vgg19_bn-c79401a0.pth'
# config['cnn']['pretrained']=False
# config = Cfg.load_config_from_name('vgg_transformer')
# # đường dẫn đến trọng số đã huấn luyện hoặc comment để sử dụng #pretrained model mặc định
# #config['weights'] = 'transformerocr.pth'
# config['device'] = 'cpu' # device chạy 'cuda:0', 'cuda:1', 'cpu'

# detector = Predictor(config)

# listText = []

# for i in range(1, len(listWord)+1):
#     img = listWord[i-1].img
#     cv2.imshow(f'Word{i}', img)
#     s = detector.predict(Image.fromarray(img), return_prob=False)
#     print('TEXT OUTPUT: ',s)
#     listText.append(s)

# # for sentence in listWord:
# #     cv2.imshow(f'Word{i}', sen)
# #     s = detector.predict(Image.fromarray(img), return_prob=False)
# #     print('TEXT OUTPUT: ',s)
#     listText.append(s)

cv2.waitKey(0)
cv2.destroyAllWindows()