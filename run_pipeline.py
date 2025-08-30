

import os
import re
import shutil
import subprocess
import sys
import threading
import time

import requests
import multiprocessing as mp
import urllib.parse

from config import update_config, _C

import filelock

# Telegram-Konfiguration
BOT_TOKEN = ""      # add your own api key here
CHAT_ID = ""       # add your own chat id here
TELEGRAM_BAD_CHARS = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
pipeline_configs_dir = "configs/pipeline"
pipeline_results_dir = "configs/pipeline_results"
first_run = True

def send_telegram_message(message):
    
    for char in TELEGRAM_BAD_CHARS:
        message = message.replace(char, f"\\{char}")

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, data=data)
    except Exception as e:
        print("Fehler beim Senden der Telegram-Nachricht:", e)


def stream_reader(stream, writer, print_prefix=""):
    for line in iter(stream.readline, ''):
        writer.write(line)
        print(print_prefix + line, end='')
    stream.close()

def run_and_log(command, log_file_path):
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          bufsize=0, universal_newlines=True) as process:
        with open(log_file_path, "w") as file_writer:
            # Start threads to read stdout and stderr concurrently
            stdout_thread = threading.Thread(target=stream_reader, args=(process.stdout, file_writer))
            stderr_thread = threading.Thread(target=stream_reader, args=(process.stderr, file_writer, "ERR: "))

            stdout_thread.start()
            stderr_thread.start()

            # Wait for process and threads to finish
            process.wait()
            stdout_thread.join()
            stderr_thread.join()
        
        return process.returncode

if __name__ == "__main__":

    if len(sys.argv) > 1:
        pipeline_configs_dir = sys.argv[1]

    if len(sys.argv) > 2:
        pipeline_results_dir = sys.argv[2]
        
    lock_file = os.path.join(pipeline_configs_dir, "lockfile.lock")

    while True:
        files_pipelined = os.listdir(pipeline_configs_dir)
        files_pipelined = sorted(files_pipelined)
        files_pipelined = [f for f in files_pipelined if f.endswith(".yaml")]

        if len(files_pipelined) == 0:
            print("No more configs to process.")
            send_telegram_message("No more configs to process.")
            break

        print("Pipeline configs found:")
        print("=====================================")

        for file in files_pipelined:
            print(f"- {file}")

        print("=====================================")

        if first_run:   # Only ask the user on the first run, otherwise the application would be blocked
            input("Press Enter to start processing the pipeline...")  # Wait for user input to start the pipeline
            first_run = False

        
        file = files_pipelined[0]
        
        with filelock.FileLock(lock_file):
            print(f"Running job {file}...")

            # Start the pipeline process
            file_path = os.path.join(pipeline_configs_dir, file)
            results_success_file_path = os.path.join(pipeline_results_dir, "success", file)
            results_failure_file_path = os.path.join(pipeline_results_dir, "failure", file)
            log_success_file_path = os.path.join(pipeline_results_dir, "success", file.replace(".yaml", ".log"))
            log_failure_file_path = os.path.join(pipeline_results_dir, "failure", file.replace(".yaml", ".log"))

            os.makedirs(os.path.dirname(results_success_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(results_failure_file_path), exist_ok=True)

            # Start the subprocess
            command = ["python", "main.py", "--config_file", file_path]
            return_code = run_and_log(command, log_success_file_path)

            print(f"Job {file} finished.")
            print(f"Return code: {return_code}")

            time.sleep(1)  # Sleep that the user can stop the pipeline if needed

            if return_code != 0:
                string_to_send = f"Job {file} failed with returncode {return_code}. {len(files_pipelined) - 1} jobs left."
                send_telegram_message(string_to_send)

            if return_code != 0:
                shutil.move(file_path, results_failure_file_path)
                shutil.move(log_success_file_path, log_failure_file_path)
            else:
                shutil.move(file_path, results_success_file_path)

        time.sleep(1)  # Sleep that the pipeline can be refilled