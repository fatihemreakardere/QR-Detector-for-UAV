import cv2
import numpy as np

def detect_qr_code(image_path):
    # Load an image from file
    image = cv2.imread(image_path)
    if image is None:
        print("Error loading image")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Optional: Apply Gaussian Blurring
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    # Thresholding the image
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter for QR code-like contours
    qr_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]  # Adjust size requirement as needed

    # Draw contours on the original image
    cv2.drawContours(image, qr_contours, -1, (0, 255, 0), 3)

    # QR Code detection and decoding
    detector = cv2.QRCodeDetector()
    data, vertices_array, _ = detector.detectAndDecode(image)
    
    if vertices_array is not None:
        # If QR Code data found, print it
        print("QR Code data:", data)
        # Draw a bounding box around the QR code
        vertices_array = np.int32(vertices_array)
        cv2.polylines(image, [vertices_array], isClosed=True, color=(0, 0, 255), thickness=5)
    else:
        print("QR Code not detected")

    # Display the image
    cv2.imshow('Image with QR code', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
detect_qr_code('i1.jpeg')
