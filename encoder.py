import numpy as np
import matplotlib.pyplot as plt

def encode_number(number):
    # Convert number to binary string (8 bits for simplicity)
    binary_data = format(number, '08b')
    print(binary_data)
    # Initialize 5x5 matrix with zeros
    matrix = np.zeros((5, 5), dtype=int)

    # Insert start and stop bits
    matrix[0, 0] = 1  # Start bit
    matrix[0, 4] = 1  # Rotation bit
    matrix[4, 0] = 1  # Stop bit

    # Insert data bits
    idx = 0
    for i in range(1, 4):
        for j in range(1, 5):
            if idx < len(binary_data):
                matrix[i, j] = int(binary_data[idx])
                idx += 1

    return matrix

def display_matrix(matrix, filename):
    # Create a color image (3 channels)
    color_matrix = np.zeros((5, 5, 3), dtype=np.uint8)
    for i in range(5):
        for j in range(5):
            if matrix[i, j] == 1:
                color_matrix[i, j] = [255, 0, 0]  # Red for 1
            else:
                color_matrix[i, j] = [0, 255, 0]  # Blue for 0

    plt.imshow(color_matrix)
    plt.axis('off')  # Hide axes
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

# Example usage
number = 42  # Number to encode
encoded_matrix = encode_number(number)
print("Encoded matrix:")
print(encoded_matrix)

# Save the plot as an image file
filename = 'images/encoded_matrix.png'
display_matrix(encoded_matrix, filename)
print(f"Encoded matrix saved as {filename}")
