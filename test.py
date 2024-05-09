import cv2
import numpy as np
from testRotateCrop import crop_rotated_rectangle
def grayscale_and_binary(image, threshold=127):
    # Đọc ảnh
    # Chuyển đổi thành ảnh xám
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Nhị phân hóa ảnh
    _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    
    return binary_image

def denoise(binary_image):
    # Sử dụng phép toán morphology để loại bỏ nhiễu
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    denoised_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    
    return denoised_image
def crop_rect(img, rect):
    # get the parameter of the small rectangle
    center, size, angle = rect[0], rect[1], rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    # get row and col num in img
    height, width = img.shape[0], img.shape[1]

    # calculate the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1)
    # rotate the original image
    img_rot = cv2.warpAffine(img, M, (width, height))

    # now rotated rectangle becomes vertical, and we crop it
    img_crop = cv2.getRectSubPix(img_rot, size, center)

    return img_crop, img_rot

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

# Đọc ảnh
image = cv2.imread('img/sample08.jpeg')
# Chuyển sang không gian màu HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Phân loại các pixel xanh dương
lower_blue = np.array([90, 50, 50])
upper_blue = np.array([130, 255, 255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)

#Thực hiện morphology để loại bỏ nhiễu
kernel = np.ones((5,5),np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = denoise(mask)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# for contour in contours:
#     # Vẽ đường viền của đối tượng
#     cv2.drawContours(image, [contour], 0, (0, 255, 0), 2)
four_sided_contours = []
for contour in contours:
    # Xấp xỉ đường viền
    x, y, w, h = cv2.boundingRect(contour)
    epsilon = 0.05 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # Kiểm tra xem contour có 4 cạnh không
    if len(approx) == 4:
        four_sided_contours.append(approx)

for i, contour in enumerate(four_sided_contours):

    x, y, w, h = cv2.boundingRect(contour)
    print("x, y, w, h: ", x, y, w, h)
    cv2.drawContours(image, [contour], 0, (0, 255, 0), 2)
    number_of_black_pix = np.sum(mask[y:y+h, x:x+w]) / 255
    if (number_of_black_pix/(w*h)) > 0.85 or (number_of_black_pix/(w*h)) < 0.2:
        print("black_pix_ratio: ", number_of_black_pix/(w*h))
        continue
    aspect_ratio = w / h
    if aspect_ratio > 3 or aspect_ratio < 0.8:
        print("aspect_ratio: ", aspect_ratio)
        continue
    # Cắt ảnh từ ảnh gốc
    if (w*h) < 2000:
        print("area: ", w*h)
        continue
    print("area: ", w*h)
    rect = cv2.minAreaRect(contour)
    print("rect: {}".format(rect))

    box = cv2.boxPoints(rect)
    print(box[0])
    print(type(box))
    box = np.int0(box)
    print(box)

    crop = four_point_transform(image, order_points(box))
    cv2.imshow(f'crop_{i}', crop)
    cv2.imwrite(f'./img/crop_img{i}.jpg', crop)
cv2.imshow('Detected Signs', image)
# Hiển thị ảnh gốc và ảnh đã xử lý
cv2.waitKey(0)
cv2.destroyAllWindows()

