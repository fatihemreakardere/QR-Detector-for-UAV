import numpy as np
import cv2

class QRDecoder:
    def preprocess_image(self, image):
        if image is None:
            raise ValueError("Failed to load image.")
        return image

    def sharpen_image(self, image):
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        sharpened = cv2.filter2D(image, -1, kernel)
        return sharpened

    def increase_contrast(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        final_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return final_image

    def find_largest_contour(self, image):
        image = self.sharpen_image(image)
        image = self.increase_contrast(image)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
            width = int(rect[1][0])
            height = int(rect[1][1])
            src_pts = box.astype("float32")
            dst_pts = np.array([[0, height - 1],
                                [0, 0],
                                [width - 1, 0],
                                [width - 1, height - 1]], dtype="float32")
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(image, M, (width, height))
            gray_matrix = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        else:
            gray_matrix = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray_matrix, image

    def enhance_contrast(self, gray_matrix):
        _, high_contrast_matrix = cv2.threshold(gray_matrix, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        high_contrast_matrix = cv2.bitwise_not(high_contrast_matrix)
        return high_contrast_matrix

    def scale_to_5x5(self, high_contrast_matrix):
        scaled_matrix = cv2.resize(high_contrast_matrix, (5, 5), interpolation=cv2.INTER_AREA)
        return scaled_matrix

    def binarize_matrix(self, scaled_matrix):
        binary_matrix = np.zeros((5, 5), dtype=int)
        for i in range(5):
            for j in range(5):
                if scaled_matrix[i, j] > 128:
                    binary_matrix[i, j] = 1
                else:
                    binary_matrix[i, j] = 0
        return binary_matrix

    def decode_number(self, binary_matrix):
        condition = binary_matrix[4, 4] == 0 and binary_matrix[0, 0] == 1 and binary_matrix[0, 4] == 1 and binary_matrix[4, 0] == 1
        if not condition:
            cnt = 0
            while not condition and cnt < 4:
                cnt += 1
                condition = binary_matrix[4, 4] == 0 and binary_matrix[0, 0] == 1 and binary_matrix[0, 4] == 1 and binary_matrix[4, 0] == 1
                if condition:
                    break
                elif cnt == 4:
                    raise ValueError("Invalid matrix format. Could not find start and stop bits.")
                else:
                    binary_matrix = np.rot90(binary_matrix)
        binary_data = ''
        for i in range(1, 3):
            for j in range(1, 5):
                binary_data += str(binary_matrix[i, j])
        number = int(binary_data, 2)
        return number