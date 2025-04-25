import customtkinter as ctk
import threading
from speech_control import listen_and_execute  # Import from separate file

# Create the main application window
app = ctk.CTk()
app.geometry("600x350")
app.title("Speech to Control Media")
app.wm_attributes("-topmost", True)

ctk.set_appearance_mode("dark")

# UI Components
pinned_frame = ctk.CTkFrame(app, height=60, fg_color="gray")
pinned_frame.pack(fill="x", pady=10)

pinned_label = ctk.CTkLabel(pinned_frame, text="Voice Control for Media", font=("Arial", 14))
pinned_label.pack(pady=10)

output_text = ctk.StringVar()
listening_label = ctk.CTkLabel(app, textvariable=output_text, font=("Arial", 14), wraplength=500)
listening_label.pack(pady=10)

result_text = ctk.StringVar()
result_label = ctk.CTkLabel(app, textvariable=result_text, font=("Arial", 12), wraplength=500)
result_label.pack(pady=20)

# Start the speech recognition in a thread
def start_thread():
    speech_thread = threading.Thread(target=listen_and_execute, args=(output_text, result_text), daemon=True)
    speech_thread.start()

start_thread()
app.mainloop()
