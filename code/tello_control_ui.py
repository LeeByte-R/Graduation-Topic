from PIL import Image
from PIL import ImageTk
import tkinter as tki
from tkinter import Toplevel, Scale
import threading
import matplotlib.pyplot as plt
import datetime
import cv2
import os
import time
import math
from tello import Tello
from darknet import load_network, detect_image, draw_boxes, load_image, array_to_image, predict_dis
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from Repeat import RepeatingTimer
import numpy as np


class TelloUI:
    def __init__(self, yolo, class_names, colors, outputpath):
        """
        Initial all the element of the GUI,support by Tkinter
        Raises:
            RuntimeError: If the Tello rejects the attempt to enter command mode.
        """
        self.yolov4 = yolo
        self.class_names = class_names
        self.colors = colors
        self.sv = False
        self.f = open('command.txt', 'w')
        self.tello = Tello('', 8889)
        self.outputPath = outputpath # the path that save pictures created by clicking the takeSnapshot button 
        self.frame = None  # frame read from h264decoder and used for pose recognition 
        self.thread = None # thread of the Tkinter mainloop
        self.stopEvent = None  
        self.sv_index = 0
        self.rotate_i = 0
        self.i = 0

        # control variables
        self.distance = 1.0  # default distance for 'move' cmd
        self.degree = 30  # default degree for 'cw' or 'ccw' cmd
        self.x = 0
        self.y = 0
        self.dis = 0
        self.print_dis = False
        # if the flag is TRUE,the auto-takeoff thread will stop waiting for the response from tello
        self.quit_waiting_flag = False
        
        # initialize the root window and image panel
        self.root = tki.Tk()
        self.panel = None

        # create buttons

        self.btn_end = tki.Button(
            self.root, text="Distance", relief="raised", command=self.predict_distance)
        self.btn_end.pack(side="bottom", fill="both",
                          expand="yes", padx=10, pady=5)

        self.btn_surrounding_view = tki.Button(
            self.root, text="Surrounding View", relief="raised", command=self.surrounding_view)
        self.btn_surrounding_view.pack(side="bottom", fill="both",
                                       expand="yes", padx=10, pady=5)

        self.btn_snapshot = tki.Button(self.root, text="Snapshot!",
                                       command=self.takeSnapshot)
        self.btn_snapshot.pack(side="bottom", fill="both",
                               expand="yes", padx=10, pady=5)

        self.btn_pause = tki.Button(self.root, text="Pause", relief="raised", command=self.pauseVideo)
        self.btn_pause.pack(side="bottom", fill="both",
                            expand="yes", padx=10, pady=5)

        self.btn_landing = tki.Button(
            self.root, text="Open Command Panel", relief="raised", command=self.openCmdWindow)
        self.btn_landing.pack(side="bottom", fill="both",
                              expand="yes", padx=10, pady=5)
        
        # start a thread that constantly pools the video sensor for
        # the most recently read frame
        self.stopEvent = threading.Event()
        self.frame_timer = RepeatingTimer(0.1, self.update_frame)
        self.frame_timer.start()
        # set a callback to handle when the window is closed
        self.root.wm_title("TELLO Controller")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)

        # the sending_command will send command to tello every 5 seconds
        # self.sending_command_thread = threading.Thread(target=self._sendingCommand)
        #self.sending_command_thread.daemon = True

    def update_frame(self):
        try:
            self.frame = self.tello.read()
            if self.frame is None or self.frame.size == 0:
                return

            self.frame = self.tello.read()
            self.predict(self.frame)

            if self.print_dis:
                cv2.putText(self.frame, str(self.dis), (int(self.x), int(self.y)), cv2.FONT_HERSHEY_SIMPLEX, 2,
                            (255, 255, 255), 5,
                            cv2.LINE_AA)

            image = Image.fromarray(self.frame)
            self._updateGUIImage(image)
        except RuntimeError as e:
            print("[INFO] caught a RuntimeError")
        except AttributeError as e:
            print("[INFO] Tello delete")

    def _updateGUIImage(self, image):
        image = ImageTk.PhotoImage(image)
        # if the panel none ,we need to initial it
        if self.panel is None:
            self.panel = tki.Label(image=image)
            self.panel.image = image
            self.panel.pack(side="left", padx=10, pady=10)
        # otherwise, simply update the panel
        else:
            self.panel.configure(image=image)
            self.panel.image = image
    """
    def _sendingCommand(self):
        while True:
            self.tello.send_command('command')        
            time.sleep(5)
    """

    def _setQuitWaitingFlag(self):
        self.quit_waiting_flag = True        
   
    def openCmdWindow(self):
        panel = Toplevel(self.root)
        panel.wm_title("Command Panel")

        self.btn_up = tki.Button(
            panel, text="up", relief="raised", command=self.telloUp)
        self.btn_up.pack(side="left", fill="both",
                         expand="yes", padx=10, pady=5)
        self.btn_down = tki.Button(
            panel, text="down", relief="raised", command=self.telloDown)
        self.btn_down.pack(side="right", fill="both",
                           expand="yes", padx=10, pady=5)

        self.btn_left = tki.Button(
            panel, text="left", relief="raised", command=self.telloMoveLeft)
        self.btn_left.pack(side="left", fill="both",
                         expand="yes", padx=10, pady=5)
        self.btn_right = tki.Button(
            panel, text="right", relief="raised", command=self.telloMoveRight)
        self.btn_right.pack(side="right", fill="both",
                            expand="yes", padx=10, pady=5)

        self.btn_forward = tki.Button(
            panel, text="forward", relief="raised", command=self.telloMoveForward)
        self.btn_forward.pack(side="left", fill="both",
                           expand="yes", padx=10, pady=5)
        self.btn_backward = tki.Button(
            panel, text="backward", relief="raised", command=self.telloMoveBackward)
        self.btn_backward.pack(side="right", fill="both",
                              expand="yes", padx=10, pady=5)

        self.btn_cw = tki.Button(
            panel, text="CW", relief="raised", command=self.telloCW)
        self.btn_cw.pack(side="left", fill="both",
                              expand="yes", padx=10, pady=5)
        self.btn_ccw = tki.Button(
            panel, text="CCW", relief="raised", command=self.telloCCW)
        self.btn_ccw.pack(side="right", fill="both",
                          expand="yes", padx=10, pady=5)

        self.btn_landing = tki.Button(
            panel, text="Land", relief="raised", command=self.telloLanding)
        self.btn_landing.pack(side="bottom", fill="both",
                              expand="yes", padx=10, pady=5)

        self.btn_takeoff = tki.Button(
            panel, text="Takeoff", relief="raised", command=self.telloTakeOff)
        self.btn_takeoff.pack(side="bottom", fill="both",
                              expand="yes", padx=10, pady=5)

        self.distance_bar = Scale(panel, from_=0.02, to=5, tickinterval=0.01, digits=3, label='Distance(m)',
                                  resolution=0.01)
        self.distance_bar.set(0.2)
        self.distance_bar.pack(side="left")

        self.btn_distance = tki.Button(panel, text="Reset Distance", relief="raised",
                                       command=self.updateDistancebar,
                                       )
        self.btn_distance.pack(side="left", fill="both",
                               expand="yes", padx=10, pady=5)

        self.degree_bar = Scale(panel, from_=1, to=360, tickinterval=10, label='Degree')
        self.degree_bar.set(30)
        self.degree_bar.pack(side="right")

        self.btn_distance = tki.Button(panel, text="Reset Degree", relief="raised", command=self.updateDegreebar)
        self.btn_distance.pack(side="right", fill="both",
                               expand="yes", padx=10, pady=5)

    def takeSnapshot(self, out=""):
        """
        save the current frame of the video as a jpg file and put it into outputpath
        """
        # grab the current timestamp and use it to construct the filename
        if out == "":
            ts = datetime.datetime.now()
            filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
            p = os.path.sep.join((self.outputPath, filename))
            cv2.imwrite(p, cv2.cvtColor(self.tello.read(), cv2.COLOR_RGB2BGR))
            print(("[INFO] saved {}".format(filename)))
        else:
            cv2.imwrite(out, cv2.cvtColor(self.tello.read(), cv2.COLOR_RGB2BGR))
            print(("[INFO] saved {}".format(out)))

    def pauseVideo(self):
        """
        Toggle the freeze/unfreze of video
        """
        if self.btn_pause.config('relief')[-1] == 'sunken':
            self.btn_pause.config(relief="raised")
            self.tello.video_freeze(False)
        else:
            self.btn_pause.config(relief="sunken")
            self.tello.video_freeze(True)

    def telloTakeOff(self):
        self.print_dis = False
        return self.tello.takeoff()                

    def telloLanding(self):
        self.print_dis = False
        return self.tello.land()

    def telloMoveForward(self):
        self.print_dis = False
        print("forward %d m" % self.distance)
        self.f.write('forward {}\n'.format(self.distance))
        return self.tello.move_forward(self.distance)

    def telloMoveBackward(self):
        self.print_dis = False
        print("backward %d m" % self.distance)
        self.f.write('backward {}\n'.format(self.distance))
        return self.tello.move_backward(self.distance)

    def telloMoveLeft(self):
        self.print_dis = False
        print("left %d m" % self.distance)
        self.f.write('left {}\n'.format(self.distance))
        return self.tello.move_left(self.distance)

    def telloMoveRight(self):
        self.print_dis = False
        print("right %d m" % self.distance)
        self.f.write('right {}\n'.format(self.distance))
        return self.tello.move_right(self.distance)

    def telloUp(self):
        self.print_dis = False
        print("up %d m" % self.distance)
        self.f.write('up {}\n'.format(self.distance))
        return self.tello.move_up(self.distance)

    def telloDown(self):
        self.print_dis = False
        print("down %d m" % self.distance)
        self.f.write('down {}\n'.format(self.distance))
        return self.tello.move_down(self.distance)

    def updateTrackBar(self):
        self.my_tello_hand.setThr(self.hand_thr_bar.get())

    def updateDistancebar(self):
        self.distance = self.distance_bar.get()
        print('reset distance to %.1f' % self.distance)

    def updateDegreebar(self):
        self.degree = self.degree_bar.get()
        print('reset distance to %d' % self.degree)

    def telloCCW(self):
        self.print_dis = False
        self.f.write('cw {}\n'.format(self.degree))
        print("ccw {} degree".format(self.degree))
        self.tello.rotate_ccw(self.degree)

    def telloCW(self):
        self.print_dis = False
        self.f.write('ccw {}\n'.format(self.degree))
        print("cw {} m".format(self.degree))
        self.tello.rotate_cw(self.degree)

    def onClose(self):
        """
        set the stop event, cleanup the camera, and allow the rest of
        the quit process to continue
        """
        #
        print("[INFO] closing...")
        self.stopEvent.set()
        self.frame_timer.cancel()
        self.root.quit()
        self.root.destroy()
        del self.tello
        self.end()

    def surrounding_view(self):
        self.print_dis = False
        print("surrounding view")
        self.rotate_i = 0
        self.f.write(f"sv {self.sv_index}\n")
        try:
            os.mkdir("SV/sv{}".format(self.sv_index))
        except FileExistsError:
            for file in os.listdir("SV/sv{}".format(self.sv_index)):
                os.remove(f"SV/sv{self.sv_index}/{file}")

        a = time.time()
        self.sv = True
        self.rotate_snap()
        print(time.time()-a)

    def rotate_snap(self):
        if self.rotate_i < 6:
            self.takeSnapshot(f"SV/sv{self.sv_index}/{self.rotate_i}.jpg")
            self.tello.rotate_cw(60)
            self.root.after(2500, self.rotate_snap)
            self.rotate_i += 1
        else:
            self.sv_index += 1
            self.rotate_i = 0

    def end(self):
        self.f.write('end')
        self.f.close()

    def predict(self, frame):
        darknet_image, _ = array_to_image(frame)
        # begin = time.time()
        predictions = detect_image(self.yolov4, self.class_names, darknet_image)
        end = time.time()
        # print(end - begin)
        # print(predictions)

        return draw_boxes(predictions, frame, self.colors)

    def predict_distance(self):
        dis = 1.0
        self.takeSnapshot("first.jpg")
        self.tello.move_backward(dis)
        self.root.after(3000, self.take_second)

    def take_second(self):
        self.takeSnapshot("second.jpg")
        dis = 1.0
        self.tello.move_forward(dis)
        self.dis, self.x, self.y = predict_dis(self.yolov4, self.class_names, self.colors, dis)
        print(f'Distance={self.dis}')
        self.f.write(f"distance {self.dis}\n")
        self.print_dis = True
