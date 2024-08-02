import socket
import struct
import cv2
import numpy as np
from picamera2 import Picamera2
from lan_config import SERVER_IP, SERVER_PORT
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../QRDecoder')))
from QRDecoder import QRDecoder


def send_frames(picam2, client_socket):
    decoder = QRDecoder()
    while True:
        frame = picam2.capture_array()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_matrix, processed_frame = decoder.find_largest_contour(frame_rgb)
        high_contrast_matrix = decoder.enhance_contrast(gray_matrix)
        try:
            
            
            scaled_matrix = decoder.scale_to_5x5(high_contrast_matrix)
            binary_matrix = decoder.binarize_matrix(scaled_matrix)
            decoded_number = decoder.decode_number(binary_matrix)
            print("Decoded number:", decoded_number)
        except ValueError as e:
            print(f"Error: {e}")
            processed_frame = frame_rgb

        _, frame_encoded = cv2.imencode('.jpg', processed_frame)
        data = frame_encoded.tobytes()
        size = len(data)
        client_socket.sendall(struct.pack(">L", size) + data)


if __name__ == '__main__':
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((f'{SERVER_IP}', SERVER_PORT))
    server_socket.listen(1)
    print("Server is listening, waiting for connection...")

    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(main={"size": (1280, 720)}))
    picam2.start()

    try:
        client_socket, client_address = server_socket.accept()
        print(f"Connection established: {client_address}")
        send_frames(picam2, client_socket)
    finally:
        picam2.stop()
        server_socket.close()
