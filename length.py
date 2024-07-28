import numpy as np
import cv2
from pyzbar.pyzbar import decode

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    return binary

def detect_qr_code(image):
    decoded_objects = decode(image)
    return decoded_objects

def get_qr_code_size(decoded_objects):
    for obj in decoded_objects:
        points = obj.polygon
        if len(points) == 4:
            pts = np.array(points, dtype=np.float32)
            rect = cv2.boundingRect(pts)
            _, _, w, h = rect
            return max(w, h)
    return None

def calculate_distance(actual_size, focal_length, image_size):
    return (actual_size * focal_length) / image_size

def process_qr_code_image(image_path, actual_size, focal_length):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image at {image_path}")
        return None
    binary_image = preprocess_image(image)
    decoded_objects = detect_qr_code(binary_image)
    image_size = get_qr_code_size(decoded_objects)
    if image_size is not None:
        distance = calculate_distance(actual_size, focal_length, image_size)
        return distance
    return None

# Example usage
image_path = r'C:\Users\MMFH\Desktop\qrcode\indir (1).png'  # Replace with your image path
actual_size = 5.0  # Actual size of the QR code in centimeters (or any consistent unit)
focal_length = 700  # Focal length of the camera in pixels (this needs to be determined experimentally)

distance = process_qr_code_image(image_path, actual_size, focal_length)
if distance:
    print(f"Estimated distance to QR code: {distance} units")
else:
    print("No QR code found or could not estimate the distance.")
