import cv2
import numpy as np


def order_points(pts):
    """
        Khởi tạo mảng chứa tọa độ của 4 điểm theo thứ tự 
        điểm trên cùng bên trái, điểm trên cùng bên phải, 
        điểm dưới cùng bên phải, điểm dưới cùng bên trái

        Args:
            pts: Mảng numpy chứa tọa độ 4 điểm
    """
    rect = np.zeros((4, 2), dtype = "float32")

    # Tính tổng tọa độ x và y của 4 điểm
    # Điểm trên cùng bên trái sẽ có tổng nhỏ nhất
    # Điểm dưới cùng bên phải sẽ có tổng lớn nhất
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    """
        Cắt ảnh theo tọa độ 4 điểm
        Args:
            image: Ảnh đầu vào
            pts: Mảng numpy chứa tọa độ 4 điểm
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # Tính toán chiều rộng của vùng cần cắt
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # Tính toán chiều cao của vùng cần cắt
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Xác định tọa độ của 4 điểm sau khi cắt
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # Lấy ma trận chuyển đổi
    M = cv2.getPerspectiveTransform(rect, dst)
    # Thực hiện chuyển đổi
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # trả về ảnh đã cắt
    return warped






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
