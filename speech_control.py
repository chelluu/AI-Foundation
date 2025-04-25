import pyautogui
import speech_recognition as sr

recognizer = sr.Recognizer()

def listen_and_execute(output_text, result_text):
    with sr.Microphone() as source:
        print("Listening for commands...")
        recognizer.adjust_for_ambient_noise(source)

        while True:
            try:
                output_text.set("Listening...")
                audio = recognizer.listen(source, timeout=5)
                output_text.set("Processing...")

                command = recognizer.recognize_google(audio).lower()
                print(f"Command recognized: {command}")
                result_text.set(f"You said: {command}")

                if command.startswith("please"):
                    command = command[6:].strip()

                repeat_count = 2 if "twice" in command else 1
                command = command.replace("twice", "").strip()

                if "play" in command or "pause" in command:
                    for _ in range(repeat_count): pyautogui.press("space")
                elif "forward" in command or "next" in command:
                    for _ in range(repeat_count): pyautogui.press("right")
                elif "rewind" in command:
                    for _ in range(repeat_count): pyautogui.press("left")
                elif "stop" in command:
                    for _ in range(repeat_count): pyautogui.press("esc")
                else:
                    print("Command not recognized.")
                    result_text.set("Command not recognized. Try again.")

            except sr.UnknownValueError:
                print("Could not understand audio.")
                result_text.set("Sorry, I couldn't understand that.")
            except sr.RequestError as e:
                print(f"Speech recognition error: {e}")
                result_text.set(f"Error: {e}")
            except Exception as e:
                print(f"An error occurred: {e}")
                result_text.set(f"Error: {str(e)}")
