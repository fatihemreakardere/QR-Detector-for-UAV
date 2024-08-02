import numpy as np
import cv2

class QRDecoder:
    def preprocess_image(self, image):
        # Check if the image was loaded successfully
        if image is None:
            raise ValueError("Failed to load image.")
        return image

    def sharpen_image(self, image):
        # Define a sharpening kernel
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        # Apply the sharpening filter to the image
        sharpened = cv2.filter2D(image, -1, kernel)
        return sharpened

    def increase_contrast(self, image):
        # Convert the image to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        # Split the LAB image to different channels
        l, a, b = cv2.split(lab)
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the L-channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        # Merge the CLAHE enhanced L-channel back with A and B channels
        limg = cv2.merge((cl, a, b))
        # Convert LAB image back to BGR
        final_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return final_image

    def find_largest_contour(self, image):
        # Sharpen the image to enhance edges
        image = self.sharpen_image(image)
        # Increase the contrast of the image
        image = self.increase_contrast(image)
        # Convert the image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Define the range for green color in HSV
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([80, 255, 255])
        # Create a mask for green color
        green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
        # Find contours in the mask
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            # Find the largest contour by area
            largest_contour = max(contours, key=cv2.contourArea)
            # Get the minimum area rectangle for the largest contour
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            # Draw the rectangle on the image
            cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
            # Get the width and height of the rectangle
            width = int(rect[1][0])
            height = int(rect[1][1])
            # Define source and destination points for perspective transformation
            src_pts = box.astype("float32")
            dst_pts = np.array([[0, height - 1],
                                [0, 0],
                                [width - 1, 0],
                                [width - 1, height - 1]], dtype="float32")
            # Get the perspective transformation matrix
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            # Warp the image to get a top-down view of the rectangle
            warped = cv2.warpPerspective(image, M, (width, height))
            # Convert the warped image to grayscale
            gray_matrix = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        else:
            # If no contours are found, convert the original image to grayscale
            gray_matrix = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray_matrix, image

    def enhance_contrast(self, gray_matrix):
        # Apply Otsu's thresholding to binarize the grayscale image
        _, high_contrast_matrix = cv2.threshold(gray_matrix, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Invert the binary image
        high_contrast_matrix = cv2.bitwise_not(high_contrast_matrix)
        return high_contrast_matrix

    def scale_to_5x5(self, high_contrast_matrix):
        # Resize the high contrast matrix to 5x5 pixels
        scaled_matrix = cv2.resize(high_contrast_matrix, (5, 5), interpolation=cv2.INTER_AREA)
        return scaled_matrix

    def binarize_matrix(self, scaled_matrix):
        # Initialize a 5x5 binary matrix
        binary_matrix = np.zeros((5, 5), dtype=int)
        # Binarize the scaled matrix based on pixel intensity
        for i in range(5):
            for j in range(5):
                if scaled_matrix[i, j] > 128:
                    binary_matrix[i, j] = 1
                else:
                    binary_matrix[i, j] = 0
        return binary_matrix

    def decode_number(self, binary_matrix):
        # Check for the presence of start and stop bits in the corners
        condition = binary_matrix[4, 4] == 0 and binary_matrix[0, 0] == 1 and binary_matrix[0, 4] == 1 and binary_matrix[4, 0] == 1
        if not condition:
            # Rotate the matrix to find the correct orientation with start and stop bits
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
        # Extract the binary data from the matrix
        binary_data = ''
        for i in range(1, 3):
            for j in range(1, 5):
                binary_data += str(binary_matrix[i, j])
        # Convert the binary string to a number
        number = int(binary_data, 2)
        return number
