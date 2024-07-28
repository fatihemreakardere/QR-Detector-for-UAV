import numpy as np
import cv2

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from path: {image_path}")

    # Display original image
    cv2.imshow('Original Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return image


def find_largest_contour(image):
    # Convert the image to binary masks for red and blue
    lower_red = np.array([0, 0, 100])
    upper_red = np.array([50, 50, 255])
    red_mask = cv2.inRange(image, lower_red, upper_red)

    lower_blue = np.array([100, 0, 0])
    upper_blue = np.array([255, 50, 50])
    blue_mask = cv2.inRange(image, lower_blue, upper_blue)

    lower_green = np.array([0, 100, 0])
    upper_green = np.array([50, 255, 50])
    green_mask = cv2.inRange(image, lower_green, upper_green)

    # Combine the masks
    binary_image = cv2.bitwise_or(red_mask, green_mask)

    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image
    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    cv2.imshow('Contours', contour_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    print(f"Found largest contour with box points: {box}")
    
    # Draw the bounding box on the original image
    cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
    cv2.imshow('Bounding Rectangle', image)
    
    # Extract the largest contour
    width = int(rect[1][0])
    height = int(rect[1][1])
    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (width, height))

    # Convert the extracted matrix to grayscale
    gray_matrix = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # Display the grayscale extracted matrix
    cv2.imshow('Extracted Grayscale Matrix', gray_matrix)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return gray_matrix


def enhance_contrast(gray_matrix):
    #Apply Otsu's thresholding
    _, high_contrast_matrix = cv2.threshold(gray_matrix, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert the matrix
    high_contrast_matrix = cv2.bitwise_not(high_contrast_matrix)

    # Display the high contrast grayscale matrix
    cv2.imshow('High Contrast Grayscale Matrix', high_contrast_matrix)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return high_contrast_matrix


def scale_to_5x5(high_contrast_matrix):
    # Resize the image to 5x5
    scaled_matrix = cv2.resize(high_contrast_matrix, (5, 5), interpolation=cv2.INTER_AREA)

    # Display the scaled 5x5 matrix
    cv2.imshow('Scaled 5x5 Matrix', scaled_matrix)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return scaled_matrix


def binarize_matrix(scaled_matrix):
    binary_matrix = np.zeros((5, 5), dtype=int)
    for i in range(5):
        for j in range(5):
            # Assuming the high contrast image results in either 0 or 255 values
            if scaled_matrix[i, j] > 128:
                binary_matrix[i, j] = 1
            else:
                binary_matrix[i, j] = 0

    # Display binarized matrix
    print("Binarized matrix:")
    print(binary_matrix)

    return binary_matrix

def rotate_matrix_90_degrees(matrix):
    # Transpose the matrix
    transposed_matrix = [list(row) for row in zip(*matrix)]
    # Reverse each row
    rotated_matrix = [row[::-1] for row in transposed_matrix]
    return rotated_matrix


def decode_number(matrix):
    print(matrix)
    condition = matrix[4,4] == 0 and matrix[0,0] == 1 and matrix[0,4] == 1 and matrix[4,0] == 1
    if not condition:
        cnt = 0
        while not condition and cnt < 4:
            cnt += 1
            condition = matrix[4,4] == 0 and matrix[0,0] == 1 and matrix[0,4] == 1 and matrix[4,0] == 1
            if condition:
                break
            elif cnt == 4:
                print("Invalid matrix format. Could not find start and stop bits.")
            else:
                matrix = np.rot90(matrix)

    binary_data = ''
    for i in range(1, 3):
        for j in range(1, 5):
            binary_data += str(matrix[i, j])
    print(f"Binary data: {binary_data}")
    number = int(binary_data, 2)
    return number


# Example usage
image_path = 'images/encoded_matrix.png'  # Path to the larger image file containing the 5x5 matrix
image = preprocess_image(image_path)
gray_matrix = find_largest_contour(image)
high_contrast_matrix = enhance_contrast(gray_matrix)
scaled_matrix = scale_to_5x5(high_contrast_matrix)
binary_matrix = binarize_matrix(scaled_matrix)
decoded_number = decode_number(binary_matrix)

print("Decoded number:", decoded_number)