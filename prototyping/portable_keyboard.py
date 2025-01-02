from pynput import keyboard

# Dictionary to track the state of keys
key_state = {}

# Function to handle key presses
def on_press(key):
    try:
        key_state[key.char] = True
    except AttributeError:  # For special keys like space, shift, etc.
        key_state[key] = True

# Function to handle key releases
def on_release(key):
    try:
        key_state[key.char] = False
    except AttributeError:
        key_state[key] = False

# Start a listener
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# Check if a key is currently pressed
def is_key_pressed(key):
    return key_state.get(key, False)