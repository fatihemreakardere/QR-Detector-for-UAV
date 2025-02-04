
# QR-Detector-for-UAV

This project is developed for the UAV team "Sarkan" to facilitate the detection and decoding of QR codes using a Raspberry Pi 4b. The project includes an encoder for encoding integer numbers into QR codes and a decoder for detecting and decoding QR codes from images or video streams.

## Project Overview

The QR-Detector-for-UAV project consists of two main components:

1. **Encoder**: Encodes integer numbers into QR codes.
2. **Decoder**: Detects and decodes QR codes from images or video streams using a Raspberry Pi 4b.

### Encoder

The encoder component is responsible for generating QR codes from integer numbers. This is useful for creating QR codes that can be placed on objects or areas of interest for the UAV to detect.

### Decoder

The main decoder algorithm script, `QRDecoder/QRDecoder.py`, is the core of the project.

## Folder Structure

- `RP4`: Contains files and scripts specific to running the project on a Raspberry Pi 4b.
- `QRDecoder`: Contains the main decoder script (`QRDecoder.py`) and associated resources.

## Getting Started

### Prerequisites

Make sure you have the following installed on your Raspberry Pi 4b:

- Python 3.7 or higher
- Required Python libraries (see `requirements.txt`)

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/fatihemreakardere/QR-Detector-for-UAV.git
    ```

2. Navigate to the project directory:

    ```bash
    cd QR-Detector-for-UAV
    ```

3. Install the required Python libraries:

    ```bash
    pip install -r requirements.txt
    ```

### Running the Decoder

To run the QR code decoder on your Raspberry Pi 4b:

1. Navigate to the `RP4` directory:

    ```bash
    cd RP4
    ```
2. Configure the LAN settings from lan_config.py.

3. Execute the decoder script:

    ```bash
    python LAN_send.py
    ```

4. Execute the reciver script wherever you want to recive output video:

    ```bash
    python LAN_recive.py
    ```

To run the QR code decoder on your PC:

1. Execute the decoder script:

    ```bash
    python decoder.py
    ```
   

The decoder will start processing the video feed or images, detecting and decoding any QR codes it encounters.

## Usage

### Encoding QR Codes

Use the encoder to generate QR codes from integer numbers. This can be done using the provided scripts or by integrating the encoder functionality into your own applications.

### Decoding QR Codes

The decoder script will automatically detect and decode QR codes from the video feed or images. The decoded information will be processed and made available for further actions or decisions by the UAV.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributions
https://github.com/jubcos
