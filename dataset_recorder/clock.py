import tkinter as tk
import time
import threading

# Function to update the time every 1ms
def update_time(label, root, stop_event):
    if stop_event.is_set():
        root.destroy()
        return
    current_time = time.strftime('%Y-%m-%d %H:%M:%S.') + str(int(time.time() * 1000) % 1000).zfill(3)
    label.config(text=current_time)
    root.after(1, update_time, label, root, stop_event)

def start_screen_clock(stop_event):
    root = tk.Tk()
    root.title("Millisecond Timestamp")
    root.geometry("300x100")
    label = tk.Label(root, font=("Helvetica", 24))
    label.pack()
    update_time(label, root, stop_event)
    root.protocol("WM_DELETE_WINDOW", lambda: stop_event.set())
    root.mainloop()

def start_clock_in_new_thread():
    stop_event = threading.Event()
    clock_thread = threading.Thread(target=start_screen_clock, args=(stop_event,))
    clock_thread.start()
    return clock_thread, stop_event
