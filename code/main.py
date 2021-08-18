import tello as tello
from tello_control_ui import TelloUI
from UI import UI
from darknet import load_network


def main():
    cfg_path = "/home/air/darknetab/cfg/yolov4-obj_30000.cfg"
    data_path = "/home/air/darknetab/data/obj.data"
    weight_path = "/home/air/darknetab/yolov4-obj_30000.weights"
    yolov4, class_names, colors = load_network(cfg_path, data_path, weight_path)
    vplayer = TelloUI(yolov4, class_names, colors, "./img/")
    # start the Tkinter mainloop
    vplayer.root.mainloop()

    ui = UI(yolov4, class_names, colors)
    ui.root.mainloop()


if __name__ == "__main__":
    main()
