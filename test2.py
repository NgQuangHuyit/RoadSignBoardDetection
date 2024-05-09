import cv2
import numpy as np

# Load ảnh
image = cv2.imread('img/sample01.jpg')
# Chuyển đổi ảnh sang không gian màu HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Thiết lập giới hạn màu xanh dương trong không gian màu HSV
lower_blue = np.array([90, 50, 50])
upper_blue = np.array([130, 255, 255])

# Tạo mặt nạ cho các điểm màu xanh dương trong phạm vi giới hạn
mask = cv2.inRange(hsv, lower_blue, upper_blue)

cv2.imshow('Mask', mask)
# Áp dụng mặt nạ vào ảnh gốc để tách biển báo xanh dương
blue_sign = cv2.bitwise_and(image, image, mask=mask)

cv2.imshow('Blue Sign', blue_sign)
# Chuyển đổi ảnh sang không gian màu xám để dễ dàng phát hiện cạnh
gray = cv2.cvtColor(blue_sign, cv2.COLOR_BGR2GRAY)

# Phát hiện cạnh trong ảnh xám
edges = cv2.Canny(gray, 50, 150)

# Tìm các contours trong ảnh và lưu trữ chúng
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Lặp qua các contours và vẽ hình chữ nhật xung quanh chúng
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    if aspect_ratio > 1.5:  # Chỉ lấy các contours có tỉ lệ rộng/cao lớn hơn một ngưỡng
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Hiển thị ảnh đã xử lý
cv2.imshow('Detected Signs', image)
cv2.waitKey(0)
cv2.destroyAllWindows()