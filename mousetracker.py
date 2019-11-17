from pynput import mouse
from pynput import keyboard
import csv
import logging
import time


print("----------------------")
print("     Mouse Tracker    ")
print("----------------------")
print("Hit the Esc key to stop recording")
participant = input("Enter a name for your participant: ")
print("Recording...")

stopwatch = time.time()
data = []


#----------------File I/O---------------------
def write_to_csv(data):
    with open(participant + '_mouse_movement_.csv', mode='w') as movement_data_file:
        movement_writer = csv.writer(movement_data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        movement_writer.writerow(['X','Y'])
        for coordinate in data:
            movement_writer.writerow(coordinate)

# logging.basicConfig(filename=("mouse_log.txt"), level=logging.DEBUG, format='%(asctime)s: %(message)s')
# logging.basicConfig(filename=(participant + "mouse_clicks.txt"), level=logging.DEBUG, format='%(asctime)s: %(message)s')
# logging.basicConfig(filename=(participant + "keystrokes.txt"), level=logging.DEBUG, format='%(asctime)s: %(message)s')




#----------------Mouse Monitoring---------------------
def on_move(x, y):
    data.append([x, y])
    # logging.info('Pointer moved to {0}'.format((x, y)))


def on_click(x, y, button, pressed):
    if pressed:
        logging.info('Mouse clicked at ({0}, {1}) with {2}'.format(x, y, button))
    # print('{0} at {1}'.format('Pressed' if pressed else 'Released', (x, y)))
    # if not pressed:
    #     # Stop listener
    #     return False


def on_scroll(x, y, dx, dy):
    logging.info('Scrolled {0} at {1}'.format('down' if dy < 0 else 'up', (x, y)))


#----------------Keyboard Monitoring---------------------

def on_press(key):
    try:
        logging.info('alphanumeric key {0} pressed'.format(
            key.char))
    except AttributeError:
        logging.info('special key {0} pressed'.format(
            key))

def on_release(key):
    logging.info('{0} released'.format(
        key))
    if key == keyboard.Key.esc:
        write_to_csv(data)
        print("Esc key pressed. Stopped recording.")
        # Stop listener
        return False


# ...or, in a non-blocking fashion:
listener = mouse.Listener(
    on_move=on_move, on_click=on_click, on_scroll=on_scroll)
listener.start()

# Collect events until released
with keyboard.Listener(
        on_press=on_press,
        on_release=on_release) as listener:
    listener.join()