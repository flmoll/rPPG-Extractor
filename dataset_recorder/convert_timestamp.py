from datetime import datetime

def to_unix_timestamp_ms(timestamp_str):
    try:
        # Parse the input timestamp
        dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")
        # Get the Unix timestamp in seconds (float)
        unix_ts = dt.timestamp()
        # Convert to milliseconds
        return int(unix_ts * 1000)
    except ValueError as e:
        print(f"Error parsing timestamp: {e}")
        return None

# Example usage:
if __name__ == "__main__":
    ts = input("Enter timestamp (YYYY-MM-DD HH:MM:SS,ffffff): ")
    print("Input timestamp:", ts)
    print(to_unix_timestamp_ms(ts))