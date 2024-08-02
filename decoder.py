from QRDecoder.QRDecoder import QRDecoder
import cv2

def main():
    decoder = QRDecoder()
    
    image_path = 'images/encoded_matrix.png'  # Path to the larger image file containing the 5x5 matrix
    image = cv2.imread(image_path)
    image = decoder.preprocess_image(image)
    gray_matrix, image = decoder.find_largest_contour(image)
    high_contrast_matrix = decoder.enhance_contrast(gray_matrix)
    scaled_matrix = decoder.scale_to_5x5(high_contrast_matrix)
    binary_matrix = decoder.binarize_matrix(scaled_matrix)
    decoded_number = decoder.decode_number(binary_matrix)

    print("Decoded number:", decoded_number)
    cv2.imshow('Bounding Rectangle', image)
    cv2.imshow('Extracted Grayscale Matrix', gray_matrix)
    cv2.imshow('High Contrast Grayscale Matrix', high_contrast_matrix)
    cv2.imshow('Scaled 5x5 Matrix', scaled_matrix)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()