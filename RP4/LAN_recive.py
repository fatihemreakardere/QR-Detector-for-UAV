import socket
import struct
import cv2
import numpy as np
from lan_config import SERVER_IP, SERVER_PORT

def receive_frames(server_ip, server_port):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_ip, server_port))
    data = b""
    payload_size = struct.calcsize(">L")

    while True:
        while len(data) < payload_size:
            packet = client_socket.recv(4096)
            if not packet:
                break
            data += packet

        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_msg_size)[0]

        while len(data) < msg_size:
            data += client_socket.recv(4096)

        frame_data = data[:msg_size]
        data = data[msg_size:]

        frame = np.frombuffer(frame_data, dtype=np.uint8)
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

        if frame is not None:
            cv2.imshow('Received Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    client_socket.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    server_ip = f'{SERVER_IP}'  # Replace with the server's IP address
    server_port = SERVER_PORT  # Replace with the server's port
    receive_frames(server_ip, server_port)
