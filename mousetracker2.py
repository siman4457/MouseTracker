import pyautogui
import csv
from pynput import keyboard
from pynput import mouse
from threading import Event, Thread


class MouseTracker:

    def __init__(self):
        self.mouse_pos_data = []
        self.num_mouse_clicks = 0
        self.num_keystrokes = 0
        self.num_scrolls = 0
        self.participant = ""
        self.timer = None


    def startup(self):
        print("----------------------")
        print("     Mouse Tracker    ")
        print("----------------------")
        print("Hit the Esc key to stop recording")
        self.participant = input("Enter a name for your participant: ")
        print("Recording...")

        # Keyboard listener in a non-blocking fashion:
        keyboard_listener = keyboard.Listener(on_release=self.on_release)
        keyboard_listener.start()

        # Mouse listener in a non-blocking fashion:
        # mouse_listener = mouse.Listener(on_click=on_click, on_scroll=on_scroll)
        mouse_listener = mouse.Listener(on_click=self.on_click)
        mouse_listener.start()


    def get_mouse_position(self):
        position = pyautogui.position()
        print("Current position:", position)
        self.mouse_pos_data.append([position.x, position.y])


    #----------------Mouse Monitoring---------------------

    def on_click(self, x, y, button, pressed):
        if pressed:
            print('Mouse clicked at ({0}, {1}) with {2}'.format(x, y, button))
            self.num_mouse_clicks += 1


    def on_scroll(self, x, y, dx, dy):
        print('Scrolled {0} at {1}'.format('down' if dy < 0 else 'up', (x, y)))
        self.num_scrolls += 1


    #----------------Keyboard Monitoring---------------------

    def on_press(self, key):
        try:
            print('alphanumeric key {0} pressed'.format(key.char))
            if key == keyboard.Key:
                self.num_keystrokes += 1

        except AttributeError:
            print('special key {0} pressed'.format(key))


    def on_release(self, key):
        if key == keyboard.Key.esc:
            self.export_to_csv(self.mouse_pos_data)
            print("Esc key pressed. Stopped recording.")
            self.timer()  # stop future calls
            # Stop listener
            return False

    def export_to_csv(self, data):
        with open(self.participant + '_mouse_movement_.csv', mode='w') as movement_data_file:
            movement_writer = csv.writer(movement_data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            movement_writer.writerow(['X','Y'])
            for coordinate in data:
                movement_writer.writerow(coordinate)

        print("Total number of mouse clicks: ", self.num_mouse_clicks)
        print("Total number of keystrokes: ", self.num_keystrokes)
        print("Total number of scrolls: ", self.num_scrolls)


def call_repeatedly( interval, func, *args):
    stopped = Event()

    def loop():
        while not stopped.wait(interval):  # the first call is in `interval` secs
            func(*args)

    Thread(target=loop).start()
    return stopped.set

def main():
    mousetracker = MouseTracker()
    mousetracker.startup()
    mousetracker.timer = call_repeatedly(1, mousetracker.get_mouse_position)





if __name__ == "__main__":
    main()