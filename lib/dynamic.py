import os
import sys
import time

class DynamicText:
    def __init__(self, text_func):
        self.text_func = text_func
    
    def __str__(self):
        return self.text_func()

class DynamicDisplay:
    frame_rate = 24
    update_interval = 1 / frame_rate
    
    def __init__(self, n_lines, extra=None, extra_height=0):
        self.n_lines = n_lines
        self.lines = [""] * n_lines
        self.extra = extra if extra else []
        self.extra_height = extra_height
        self.__cls = "cls" if sys.platform.startswith("win") else "clear"
        self.__last_update = time.time()
        self.__worker_thread = None
        self.stoped = False

    def __enter__(self):
        self.__update_loop()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # self.clear_screen()
        time.sleep(self.update_interval * 2)
        self.stoped = True
        self.__worker_thread.join()

    def clear_screen(self):
        os.system(self.__cls)  # Windows

    def __update_loop(self):
        import threading
        self.__worker_thread = threading.Thread(target=self.update_lines)
        self.__worker_thread.start()

    def update_lines(self):
        while True:
            if self.stoped:
                break
            time_now = time.time()
            if time_now - self.__last_update < self.update_interval:
                continue
            self.__last_update = time_now
            self.clear_screen()
            for line in self.lines:
                print(str(line), file=sys.stdout)
            if self.extra_height > 0 and len(self.extra) > 0:
                for extra_line in (self.extra[0] if len(self.extra[0]) <= self.extra_height else self.extra[0][-self.extra_height:]):
                    print(extra_line, file=sys.stdout)
            sys.stdout.flush()

    def set_line(self, index, message):
        if 0 <= index < self.n_lines:
            self.lines[index] = message
        else:
            raise IndexError("Index out of range")
