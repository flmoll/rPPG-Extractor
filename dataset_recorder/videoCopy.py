import subprocess
import os

# List of common video folders to search
VIDEO_DIRS = [
    "/sdcard/DCIM/Camera",
    "/sdcard/Movies",
    "/sdcard/Download",
    "/sdcard/WhatsApp/Media/WhatsApp Video",
]
adb_command = 'adb'
windows_temp_file = 'C:\\Users\\flori\\Downloads\\video.mp4'

windows_temp_file = os.path.normpath(windows_temp_file)

# File extensions to look for
VIDEO_EXTENSIONS = [".mp4", ".3gp", ".mkv", ".webm"]

def list_videos_in_dir(dir_path):
    result = subprocess.run(
        [adb_command, "shell", f"ls -t '{dir_path}'"],
        stdin=subprocess.DEVNULL,  # otherwise the input reading will fail
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        return []
    files = result.stdout.strip().splitlines()
    files = [f for f in files if any(f.lower().endswith(ext) for ext in VIDEO_EXTENSIONS)]
    return [dir_path + "/" + f for f in files]

def find_newest_video():
    all_videos = []
    for dir_path in VIDEO_DIRS:
        videos = list_videos_in_dir(dir_path)
        all_videos.extend(videos)

    if not all_videos:
        print("❌ No video files found.")
        return None

    # Get modification timestamps
    def get_mtime(filepath):
        result = subprocess.run(
            [adb_command, "shell", f"stat -c %Y '{filepath}'"],
            stdin=subprocess.DEVNULL,  # otherwise the input reading will fail
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            return 0
        return int(result.stdout.strip())

    all_videos_with_time = [(get_mtime(f), f) for f in all_videos]
    newest = max(all_videos_with_time, key=lambda x: x[0])
    return newest[1]

def pull_video(phone_path, local_path):
    print(f"📥 Pulling '{phone_path}' → '{windows_temp_file}'")

    subprocess.run(
        [adb_command, "pull", phone_path, windows_temp_file],
        stdin=subprocess.DEVNULL  # otherwise the input reading will fail
    )

    if os.path.exists(local_path):
        ans = input(f"File '{local_path}' already exists. Press enter to continue")
        os.remove(local_path)

    os.rename(windows_temp_file, local_path)
    print("✅ Done.")