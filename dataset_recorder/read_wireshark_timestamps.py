import re
import subprocess
import numpy as np
import pyshark

import matplotlib.pyplot as plt

tshark_executable = "C:\\Program Files\\Wireshark\\tshark"


def read_usb_packets_hexdump(pcap_file):
    """
    Runs tshark -x to get full packet hex dump and extracts the raw bytes per packet.
    Returns a list of bytes objects (one per packet).
    """

    try:
        # Run tshark -x to get hex dump of each packet
        proc = subprocess.run(
            [tshark_executable, "-r", pcap_file, "-x"],
            capture_output=True,
            text=True,
            check=True
        )

        output = proc.stdout

        # tshark prints packets like:
        # 0000  xx xx xx xx xx xx xx xx  xx xx xx xx xx xx xx xx  |................|
        # 0010  xx xx xx xx xx xx xx xx  xx xx xx xx xx xx xx xx  |................|
        # ...
        # followed by empty line before next packet dump.

        # Split output into packets (by empty lines)
        raw_packets = output.strip().split('\n\n')

        packets_bytes = []

        for packet_dump in raw_packets:
            # Extract all hex bytes ignoring offset and ASCII part
            hex_lines = packet_dump.splitlines()
            hex_bytes = []
            for line in hex_lines:
                bytes_on_line = re.findall(r'(?:^|\s)([0-9a-f]{2})(?=\s|$)', line)  # add leading space to catch first byte
                hex_bytes.extend(bytes_on_line)

            if not hex_bytes:
                continue

            packets_bytes.append(bytes.fromhex(''.join(hex_bytes)))

        return packets_bytes

    except subprocess.CalledProcessError as e:
        print(f"Error running tshark: {e}")
        print(f"stderr: {e.stderr}")
        return []
    except FileNotFoundError:
        print("tshark executable not found. Make sure tshark is installed and in your PATH.")
        return []
    
def is_ppg_sample_packet(packet, data):
    if packet.highest_layer != "DATA":
        return False
    if int(packet.length) != 91:
        return False
    if len(data) != 36:
        return False
    if data[0:2] != "eb" or data[12:14] != "eb" or data[24:26] != "eb":
        return False
    return True

def read_wireshark_timestamps(pcap_file):
    """
    Reads timestamps from a Wireshark PCAP file.

    Args:
        pcap_file (str): Path to the PCAP file.

    Returns:
        list: List of timestamps in milliseconds.
    """
    
    usb_data = read_usb_packets_hexdump(pcap_file)

    try:
        capture = pyshark.FileCapture(pcap_file, display_filter='frame.time')
        timestamps = []
        indexes = []
        datas = []

        for packet in capture:
            print(f"Processing packet: {packet.number}")

            curr_idx = int(packet.number) - 1
            curr_data = usb_data[curr_idx].hex()[54:90]  # Extract the relevant bytes from the packet data

            if not is_ppg_sample_packet(packet, curr_data):
                continue

            datas.append(curr_data)

            indexes.append(curr_idx)  # packet index
            timestamps.append(float(packet.sniff_timestamp))  # timestamp as a string

        capture.close()
        return timestamps, indexes, datas

    except Exception as e:
        print(f"Error reading PCAP file: {e}")
        return None
    
def get_ppg_from_wireshark_recording(pcap_file):
    """
    Extracts PPG data from a Wireshark recording.

    Args:
        pcap_file (str): Path to the PCAP file.

    Returns:
        list: List of PPG data samples.
    """
    timestamps, indexes, datas = read_wireshark_timestamps(pcap_file)

    assert len(timestamps) == len(indexes) and len(timestamps) == len(datas), "Timestamps, indexes, and datas must have the same length."

    if timestamps is None:
        print("No timestamps found or an error occurred.")

    timestamps = np.array(timestamps, dtype=np.float64)  # Convert timestamps to numpy array with float64 type
    timestamps = (timestamps * 1000).astype(np.uint64)
    indexes = np.array(indexes)
    #diff = np.diff(timestamps) * 1000  # Convert to milliseconds
    #diff = np.insert(diff, 0, 0)  # Insert a zero at the beginning to match the length of indexes

    data_samples = []

    for curr_data in datas:
        matches = re.findall(r'eb([0-9a-f]{10})', curr_data, re.IGNORECASE)
        uint32_values = [int(sample, 16) for sample in matches]
        if len(uint32_values) != 3:
            print(f"Warning: Expected 3 samples, found {len(uint32_values)} in data: {curr_data}")
            continue
        data_samples.append(uint32_values)

    data_samples = np.array(data_samples, dtype=np.uint64)
    print(f"Found {data_samples.shape} PPG data samples.")
    # Apply bitmask to every sample (e.g., keep only lower 24 bits)
    #status_bits =  data_samples & 0xFFFF000000 >> 24
    data_samples = data_samples & 0x0000FFFFFF

    data_samples = np.mean(data_samples.reshape(-1, 3), axis=1)  # Average every 3 samples
    return data_samples, timestamps
    
if __name__ == "__main__":
    pcap_file = "C:\\Users\\flori\\DATA\\Uni\\MotionMagnification\\Code\\dataset_recorder\\data\\filmklappe\\recording.pcapng"
    
    data_samples, timestamps = get_ppg_from_wireshark_recording(pcap_file)

    plt.plot(timestamps, data_samples, label='PPG Data Samples')
    plt.xlabel('Sample Index')
    plt.ylabel('PPG Data Value')
    plt.title('PPG Data Samples from Wireshark Timestamps')
    plt.legend()
    plt.show()
    
    """
    plt.plot(indexes, diff, label='Time Difference (ms)')
    #plt.plot(indexes[ppg_idx], diff[ppg_idx], 'ro', label='PPG Peaks')
    plt.xlabel('Packet Index')
    plt.ylabel('Time Difference (ms)')
    plt.title('Wireshark Packet Time Differences')
    plt.legend()
    plt.show()

    
    cumsum_diff = np.cumsum(diff)
    recurring_sample_rate = (3 * np.arange(len(diff))) / (cumsum_diff / 1000)  # Convert to Hz

    print("avg sampling rate: ", 3*len(diff) / (np.sum(diff) / 1000), "Hz")

    plt.plot(indexes, recurring_sample_rate, label='Recurring Sample Rate (Hz)')
    plt.xlabel('Packet Index')
    plt.ylabel('Sample Rate (Hz)')
    plt.title('Recurring Sample Rate from Wireshark Timestamps')
    plt.legend()
    plt.show()
    """