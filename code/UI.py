import tkinter as tk
from PIL import ImageTk, Image
import os
import math
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from darknet import detect_image, draw_boxes, array_to_image, bbox2points
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


class UI:
    def __init__(self, yolo, class_names, colors):
        self.yolov4 = yolo
        self.class_names = class_names
        self.colors = colors
        self.get_map()
        self.n = 6
        self.sv_i = 0
        self.i = 0
        self.original_i = 0
        try:
            self.f = open("SV/sv0/data.txt", "r")
            self.data = dict()
            lines = self.f.readlines()
            for line in lines:
                a = line.strip("\n").split("=")
                self.data[a[0]] = a[1]
        except FileNotFoundError:
            self.data = {"person": 0, "dog": 0, "cat": 0}
            self.sv_i = -1

        self.root = tk.Tk()

        self.topf = tk.Frame(self.root)
        self.topf.pack(side="top")
        self.bottomf = tk.Frame(self.root)
        self.bottomf.pack(side="bottom")

        self.input_s = tk.StringVar()
        self.input_s.set("0")
        self.input = tk.Entry(self.bottomf, width=20, textvariable=self.input_s)
        self.input.pack(side="left")

        self.btn_enter = tk.Button(self.bottomf, text='enter', fg='blue', command=self.update_sv_i)
        self.btn_enter.pack(side="left")

        self.btn_forward = tk.Button(self.bottomf, text='<<-', fg='green', command=self.forward_image)
        self.btn_forward.pack(side="left")

        self.btn_view = tk.Button(self.bottomf, text='surrounding view', fg='red', relief="raised",
                                  command=self.surrounding_view)
        self.btn_view.pack(side="left")

        self.btn_back = tk.Button(self.bottomf, text='->>', fg='green', command=self.backward_image)
        self.btn_back.pack(side="left")

        self.text = tk.StringVar()
        self.text.set(f"person={self.data['person']} dog={self.data['dog']} cat={self.data['cat']}")
        self.data_label = tk.Label(self.bottomf, textvariable=self.text)
        self.data_label.pack(side="left")

        imgg = Image.open("foo.png")
        imgg = imgg.resize((640, 480), Image.ANTIALIAS)
        self.img = ImageTk.PhotoImage(imgg)
        self.panel = tk.Label(self.topf, image=self.img)

        self.svpanel = None
        if self.sv_i < 0:
            image = Image.open("blank.jpg")
        else:
            image = Image.open("SV/sv0/yolo-0.jpg")
        image = image.resize((640, 480), Image.ANTIALIAS)
        self.update_image(image)

        self.panel.pack(side="left", padx=10, pady=10)

    def surrounding_view(self):
        if int(self.sv_i) < 0:
            return
        self.original_i = self.i
        self.image_view()

    def image_view(self):
        path = f"SV/sv{self.sv_i}"
        image = Image.open(f"{path}/yolo-{self.i}.jpg")
        image = image.resize((640, 480), Image.ANTIALIAS)
        # print(f"{path}/{self.i}.jpg")
        self.update_image(image)
        self.i += 1
        if self.i >= self.n:
            self.i = 0
        if self.i != self.original_i:
            self.root.after(100, self.image_view)
        else:
            time.sleep(0.1)
            image = Image.open(f"{path}/yolo-{self.i}.jpg")
            image = image.resize((640, 480), Image.ANTIALIAS)
            print(f"{path}/{self.i}.jpg")
            self.update_image(image)

    def update_image(self, image):
        image = ImageTk.PhotoImage(image)
        # if the panel none ,we need to initial it
        if self.svpanel is None:
            self.svpanel = tk.Label(self.topf, image=image)
            self.svpanel.image = image
            self.svpanel.pack(side="right", padx=10, pady=10)
        # otherwise, simply update the panel
        else:
            self.svpanel.configure(image=image)
            self.svpanel.image = image

    def update_sv_i(self):
        s = self.input_s.get()
        if not s.isnumeric() or int(s) >= len(os.listdir("SV")):
            s = "0"
            self.input_s.set("0")
        self.sv_i = s

        image = Image.open(f"SV/sv{self.sv_i}/yolo-0.jpg")
        image = image.resize((640, 480), Image.ANTIALIAS)
        self.update_image(image)
        self.f = open(f"SV/sv{self.sv_i}/data.txt", "r")
        self.data = dict()
        lines = self.f.readlines()
        for line in lines:
            a = line.strip("\n").split("=")
            self.data[a[0]] = a[1]
        self.text.set(f"person={self.data['person']} dog={self.data['dog']} cat={self.data['cat']}")

    def forward_image(self):
        if int(self.sv_i) < 0:
            return
        self.i -= 1
        path = f"SV/sv{self.sv_i}"
        if self.i < 0:
            self.i = self.n - 1
        path = f"SV/sv{self.sv_i}"
        image = Image.open(f"{path}/yolo-{self.i}.jpg")
        image = image.resize((640, 480), Image.ANTIALIAS)
        self.update_image(image)

    def backward_image(self):
        if int(self.sv_i) < 0:
            return
        self.i += 1
        path = f"SV/sv{self.sv_i}"
        if self.i >= self.n:
            self.i = 0
        path = f"SV/sv{self.sv_i}"
        image = Image.open(f"{path}/yolo-{self.i}.jpg")
        image = image.resize((640, 480), Image.ANTIALIAS)
        self.update_image(image)

    def get_map(self):
        x = [0]
        y = [0]
        px = [0]
        py = [0]
        sv_x = []
        sv_y = []
        fire = []
        angle = 0

        with open('command.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                command = line.strip("\n").split(" ")
                if command[0] == 'end':
                    break
                elif command[0] == 'forward':
                    d = float(command[1])
                    sin = math.sin(math.radians(angle))
                    cos = math.cos(math.radians(angle))
                    x.append(sin * d + x[-1])
                    y.append(cos * d + y[-1])
                elif command[0] == 'backward':
                    d = float(command[1])
                    sin = math.sin(math.radians(angle))
                    cos = math.cos(math.radians(angle))
                    x.append(-sin * d + x[-1])
                    y.append(-cos * d + y[-1])
                elif command[0] == 'left':
                    d = float(command[1])
                    sin = math.sin(math.radians(angle))
                    cos = math.cos(math.radians(angle))
                    x.append(-cos * d + x[-1])
                    y.append(sin * d + y[-1])
                elif command[0] == 'right':
                    d = float(command[1])
                    sin = math.sin(math.radians(angle))
                    cos = math.cos(math.radians(angle))
                    x.append(cos * d + x[-1])
                    y.append(-sin * d + y[-1])
                elif command[0] == 'cw':
                    angle += int(command[1])
                elif command[0] == 'ccw':
                    angle -= int(command[1])
                elif command[0] == 'sv':
                    sv_x.append(x[-1])
                    sv_y.append(y[-1])
                    fire.append(self.fire_size(command[1]))
                elif command[0] == 'distance':
                    d = float(command[1])
                    sin = math.sin(math.radians(angle))
                    cos = math.cos(math.radians(angle))
                    px.append(sin * d + x[-1])
                    py.append(cos * d + y[-1])

        for i in range(len(sv_x)):
            plt.scatter(sv_x[i], sv_y[i], c=fire[i], s=200, zorder=1)
            plt.text(sv_x[i] - 0.03, sv_y[i] - 0.03, '{}'.format(i))

        for i in range(1, len(px)):
            self.add_person(px[i], py[i], zoom=0.02)

        for i in range(len(x)):
            plt.plot(x[i:i + 2], y[i:i + 2], 'b-', linewidth=10, zorder=-10)

        a = max(max(x), max(y), max(px), max(py)) + 1
        b = min(min(x), min(y), min(px), min(py)) - 1
        plt.xlim(b, a)
        plt.ylim(b, a)

        if x[0] == x[-1] and y[0] == y[-1]:
            plt.text(x[0] + 0.12, y[0] + 0.12, "START")
            self.add_tello(x[0], y[0], 0.06)
        else:
            plt.text(x[0] + 0.12, y[0] + 0.12, "START")
            self.add_tello(x[0], y[0], 0.06)
            plt.text(x[-1] + 0.12, y[-1] + 0.12, "END")
            self.add_tello(x[-1], y[-1], 0.06)
        plt.savefig('foo.png')
        print("save")

    def fire_size(self, index):
        path = f"SV/sv{index}/"
        file = open(f"{path}data.txt", "w")
        count = 0
        fire = "green"
        person = 0
        dog = 0
        cat = 0
        for i in range(6):
            image = cv2.cvtColor(cv2.imread(f"{path}{i}.jpg"), cv2.COLOR_BGR2RGB)
            print(f"{path}{i}.jpg")
            darknet_image, _ = array_to_image(image)
            predictions = detect_image(self.yolov4, self.class_names, darknet_image)
            print(predictions)
            image_box = draw_boxes(predictions, image, self.colors)
            cv2.imwrite(f"{path}yolo-{i}.jpg", cv2.cvtColor(image_box, cv2.COLOR_RGB2BGR))
            zero = np.zeros((720, 960))
            for a in predictions:
                if a[0] == "fire" or a[0] == "smoke":
                    x, y, w, h = a[2]
                    xmin = max(int(round(x - (w / 2))), 0)
                    xmax = max(int(round(x + (w / 2))), 0)
                    ymin = max(int(round(y - (h / 2))), 0)
                    ymax = max(int(round(y + (h / 2))), 0)
                    # print(ymin, ymax, xmin, xmax)
                    zero[ymin:ymax, xmin:xmax] = 1
                elif a[0] == "person":
                    person += 1
                elif a[0] == "dog":
                    dog += 1
                elif a[0] == "cat":
                    cat += 1
                print(np.sum(zero))
            count += np.sum(zero)

        percent = count / (960 * 720 * 6)
        print(percent)
        if 0 < percent < 0.25:
            fire = "orange"
        elif 0.25 <= percent <= 0.75:
            fire = "orangered"
        elif 0.75 < percent:
            fire = "red"

        file.write(f"fire={fire}\n")
        file.write(f"person={person}\n")
        file.write(f"dog={dog}\n")
        file.write(f"cat={cat}")
        # file.write(f"percent={percent}")
        file.close()
        return fire

    def add_person(self, x, y, zoom=1.0):
        ax = plt.gca()
        image = plt.imread("person.jpg")
        im = OffsetImage(image, zoom=zoom)
        ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=False)
        ax.add_artist(ab)

    def add_tello(self, x, y, zoom=1.0):
        ax = plt.gca()
        image = plt.imread("tello.jpg")
        im = OffsetImage(image, zoom=zoom)
        ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=False)
        ab.zorder = -5
        ax.add_artist(ab)

