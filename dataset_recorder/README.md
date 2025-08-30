# Emergency Videos Recorder

This project provides tools for recording the realistic videos dataset as described in the paper.

## Features

- Fast video recording setup
- Configurable recording parameters
- Automatic file management
- Compatible with Windows 11

## Getting Started

### Prerequisites

- Python 3 and pip3
- Git
- A pulse oxymeter and a mobile phone (tested with Beurer PO80 and Google Pixel 8)
- Wireshark should be installed
- ADB tools should be installed

### Setup

- Connect the pulse oxymeter to a finger
- Connect the mobile phone to the PC via USB cable and enable USB Debugging in the developer options
- Test the connection to the phone with adb devices

### Installation

Clone the repository:

```bash
git clone <repository-url>
cd dataset_recorder
```

Install Python dependencies:

```bash
pip3 install -r requirements.txt
```

### Usage

Run the recorder:

```bash
python3 recorder.py
```

Then follow the instructions on screen. The recorded data will be saved into the directory data. A subdirectory is created for every recording.
When the Wireshark recording is finished save it to recording.pcapng into the newly created directory.

## License

See [LICENSE](../LICENSE) for details.
