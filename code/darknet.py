#!python3
"""
Python 3 wrapper for identifying objects in images

Requires DLL compilation

Both the GPU and no-GPU version should be compiled; the no-GPU version should be renamed "yolo_cpp_dll_nogpu.dll".

On a GPU system, you can force CPU evaluation by any of:

- Set global variable DARKNET_FORCE_CPU to True
- Set environment variable CUDA_VISIBLE_DEVICES to -1
- Set environment variable "FORCE_CPU" to "true"
- Set environment variable "DARKNET_PATH" to path darknet lib .so (for Linux)

Directly viewing or returning bounding-boxed images requires scikit-image to be installed (`pip install scikit-image`)

Original *nix 2.7: https://github.com/pjreddie/darknet/blob/0f110834f4e18b30d5f101bf8f1724c34b7b83db/python/darknet.py
Windows Python 2.7 version: https://github.com/AlexeyAB/darknet/blob/fc496d52bf22a0bb257300d3c79be9cd80e722cb/build/darknet/x64/darknet.py

@author: Philip Kahn
@date: 20180503
"""
from ctypes import *
import math
import random
import os
import cv2
import argparse
import imutils
from imutils import perspective
from imutils import contours
import sys
import numpy as np


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int),
                ("uc", POINTER(c_float)),
                ("points", c_int),
                ("embeddings", POINTER(c_float)),
                ("embedding_size", c_int),
                ("sim", c_float),
                ("track_id", c_int)]

class DETNUMPAIR(Structure):
    _fields_ = [("num", c_int),
                ("dets", POINTER(DETECTION))]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


def network_width(net):
    return lib.network_width(net)


def network_height(net):
    return lib.network_height(net)


def bbox2points(bbox):
    """
    From bounding box yolo format
    to corner points cv2 rectangle
    """
    x, y, w, h = bbox
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def class_colors(names):
    """
    Create a dict with one random BGR color for each
    class name
    """
    return {name: (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)) for name in names}


def load_network(config_file, data_file, weights, batch_size=1):
    """
    load model description and weights from config files
    args:
        config_file (str): path to .cfg model file
        data_file (str): path to .data model file
        weights (str): path to weights
    returns:
        network: trained model
        class_names
        class_colors
    """
    network = load_net_custom(
        config_file.encode("ascii"),
        weights.encode("ascii"), 0, batch_size)
    metadata = load_meta(data_file.encode("ascii"))
    class_names = [metadata.names[i].decode("ascii") for i in range(metadata.classes)]
    colors = class_colors(class_names)
    return network, class_names, colors


def print_detections(detections, coordinates=False):
    print("\nObjects:")
    for label, confidence, bbox in detections:
        x, y, w, h = bbox
        if coordinates:
            print("{}: {}%    (left_x: {:.0f}   top_y:  {:.0f}   width:   {:.0f}   height:  {:.0f})".format(label, confidence, x, y, w, h))
        else:
            print("{}: {}%".format(label, confidence))


def draw_boxes(detections, image, colors):
    for label, confidence, bbox in detections:
        left, top, right, bottom = bbox2points(bbox)
        cv2.rectangle(image, (left, top), (right, bottom), colors[label], 10)
        cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),
                    (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colors[label], 2)
    return image


def decode_detection(detections):
    decoded = []
    for label, confidence, bbox in detections:
        confidence = str(round(confidence * 100, 2))
        decoded.append((str(label), confidence, bbox))
    return decoded


def remove_negatives(detections, class_names, num):
    """
    Remove all classes with 0% confidence within the detection
    """
    predictions = []
    for j in range(num):
        for idx, name in enumerate(class_names):
            if detections[j].prob[idx] > 0:
                bbox = detections[j].bbox
                bbox = (bbox.x, bbox.y, bbox.w, bbox.h)
                predictions.append((name, detections[j].prob[idx], (bbox)))
    return predictions


def detect_image(network, class_names, image, thresh=.5, hier_thresh=.5, nms=.45):
    """
        Returns a list with highest confidence class and their bbox
    """
    pnum = pointer(c_int(0))
    predict_image(network, image)
    detections = get_network_boxes(network, image.w, image.h,
                                   thresh, hier_thresh, None, 0, pnum, 0)
    num = pnum[0]
    if nms:
        do_nms_sort(detections, num, len(class_names), nms)
    predictions = remove_negatives(detections, class_names, num)
    predictions = decode_detection(predictions)
    free_detections(detections, num)
    return sorted(predictions, key=lambda x: x[1])


def array_to_image(arr):
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2,0,1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w,h,c,data)
    return im, arr


#  lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
#  lib = CDLL("libdarknet.so", RTLD_GLOBAL)
hasGPU = True
if os.name == "nt":
    cwd = os.path.dirname(__file__)
    os.environ['PATH'] = cwd + ';' + os.environ['PATH']
    winGPUdll = os.path.join(cwd, "yolo_cpp_dll.dll")
    winNoGPUdll = os.path.join(cwd, "yolo_cpp_dll_nogpu.dll")
    envKeys = list()
    for k, v in os.environ.items():
        envKeys.append(k)
    try:
        try:
            tmp = os.environ["FORCE_CPU"].lower()
            if tmp in ["1", "true", "yes", "on"]:
                raise ValueError("ForceCPU")
            else:
                print("Flag value {} not forcing CPU mode".format(tmp))
        except KeyError:
            # We never set the flag
            if 'CUDA_VISIBLE_DEVICES' in envKeys:
                if int(os.environ['CUDA_VISIBLE_DEVICES']) < 0:
                    raise ValueError("ForceCPU")
            try:
                global DARKNET_FORCE_CPU
                if DARKNET_FORCE_CPU:
                    raise ValueError("ForceCPU")
            except NameError as cpu_error:
                print(cpu_error)
        if not os.path.exists(winGPUdll):
            raise ValueError("NoDLL")
        lib = CDLL(winGPUdll, RTLD_GLOBAL)
    except (KeyError, ValueError):
        hasGPU = False
        if os.path.exists(winNoGPUdll):
            lib = CDLL(winNoGPUdll, RTLD_GLOBAL)
            print("Notice: CPU-only mode")
        else:
            # Try the other way, in case no_gpu was compile but not renamed
            lib = CDLL(winGPUdll, RTLD_GLOBAL)
            print("Environment variables indicated a CPU run, but we didn't find {}. Trying a GPU run anyway.".format(winNoGPUdll))
else:
    lib = CDLL("/home/air/darknetab/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

copy_image_from_bytes = lib.copy_image_from_bytes
copy_image_from_bytes.argtypes = [IMAGE,c_char_p]

predict = lib.network_predict_ptr
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

if hasGPU:
    set_gpu = lib.cuda_set_device
    set_gpu.argtypes = [c_int]

init_cpu = lib.init_cpu

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_batch_detections = lib.free_batch_detections
free_batch_detections.argtypes = [POINTER(DETNUMPAIR), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict_ptr
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

load_net_custom = lib.load_network_custom
load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
load_net_custom.restype = c_void_p

free_network_ptr = lib.free_network_ptr
free_network_ptr.argtypes = [c_void_p]
free_network_ptr.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

predict_image_letterbox = lib.network_predict_image_letterbox
predict_image_letterbox.argtypes = [c_void_p, IMAGE]
predict_image_letterbox.restype = POINTER(c_float)

network_predict_batch = lib.network_predict_batch
network_predict_batch.argtypes = [c_void_p, IMAGE, c_int, c_int, c_int,
                                   c_float, c_float, POINTER(c_int), c_int, c_int]
network_predict_batch.restype = POINTER(DETNUMPAIR)


def distance(network, class_names, colors, img):
    bool = False
    # read image
    image = cv2.imread(img)
    darknet_image, _ = array_to_image(image)
    pre = detect_image(network, class_names, darknet_image)
    r = []
    for a in pre:
        if a[0] == 'person':
            bool = True
            r.append(list(a))
    x_list = []
    #print(image.shape[1])
    for i in range(len(r)):
        x_list.append(round(abs(r[i][2][0]-((image.shape[1])/2))))
    # print("r:", r)
    # print(f"x={x_list}")
    number = x_list.index(min(x_list))
    # print(f"n={number}")
    #catch x,y,w,h
    #print(r)
    r1 = []
    r2 = []
    r1 = r[number]
    r2 = r1[2]
    yolo_x1 = r2[0] - r2[2] / 2
    yolo_x2 = r2[0] + r2[2] / 2
    yolo_y1 = r2[1] - r2[3] / 2
    yolo_y2 = r2[1] + r2[3] / 2
    # print(yolo_x1,yolo_x2,yolo_y1,yolo_y2)
    print("r2:", r2[2])

    #yolo catch
    crop = image[int(yolo_y1)-10:int(yolo_y2)+10, int(yolo_x1)-10:int(yolo_x2)+10]

    #opencv catch
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    #gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(gray, 30, 60)
    kernel1 = np.ones((1, 1), np.uint8)
    kernel2 = np.ones((3, 3), np.uint8)
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closing = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel3)
    eroded = cv2.erode(closing, kernel1, iterations=1)
    dilated = cv2.dilate(eroded, kernel2, iterations=1)
    #cv2.imwrite("edged" +img , edged)
    #cv2.imwrite("edged_closing" + img , closing)
    #cv2.imwrite("dilated_" + img , dilated)
    #cv2.imwrite("eroded_" + img , eroded)
    cnts = cv2.findContours(dilated.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    (cnts, _) = contours.sort_contours(cnts)

    alist = [0]
    maxwidth = max(alist)
    count = 0
    for c in cnts:
        count += 1
        if cv2.contourArea(c) < 10000:
            continue
        orig = crop.copy()

        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 255, 0), 5)

        if w < 1 / 2 * r2[2]:
            continue
        #if w > 2/3 * r2[2]:
        #    alist.append(w)
        #maxwidth = sum(alist)/(len(alist)-1)
        alist.append(w)
        #cv2.namedWindow('image',cv2.WINDOW_NORMAL)
        #cv2.imshow("image", orig)
        #cv2.waitKey(0)
        #cv2.imwrite("opencv_b/"+str(count)+img, orig)
        # print("opencv succed!")

    maxwidth = max(alist)
    alist.append(int(r2[2])+5)
    # print("alist:",alist)
    if maxwidth > alist[-1] or maxwidth == 0:
        maxwidth = alist[-1]

    # print("max:",maxwidth)
    if maxwidth == 0:
        maxwidth=r2[2]
    point1 = (int(yolo_x1 - 20), int(yolo_y1 - 20))
    point2 = (int(yolo_x2 + 20), int(yolo_y2 + 20))
    picture = cv2.rectangle(image, point1, point2, (0, 0, 255), 5)
    cv2.imwrite(f"yolo-{img}", picture)
    return bool, maxwidth, int(yolo_x1 - 40), int(yolo_y1 - 40)


def distance_cal(width1, x1, y1, width2, x2, y2, dis):
    # 找出兩檔案最大值
    # 計算
    # dis=(float(dis)*int(width2))/(int(width2)-int(width1)) #飛近
    #print(f"w1={width1}")
    #print(f"w2={width2}")
    if int(width1) - int(width2) == 0:
        print("distance is same")
        dis = 0
    else:
        dis = (float(dis) * int(width2)) / (int(width1) - int(width2))  # 飛遠
        dis = round(dis, 3)
    print("dis:", dis)
    # picture = cv2.imread("yolo-second.jpg")
    # cv2.putText(picture, str(dis), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5,
    #             cv2.LINE_AA)
    # cv2.imwrite("final.jpg", picture)
    return dis, x1, y1


def predict_dis(network, class_names, colors, dis):
    p1, w1, x1, y1 = distance(network, class_names, colors, "first.jpg")
    p2, w2, x2, y2 = distance(network, class_names, colors, "second.jpg")
    # print("w1:", w1)
    # print("w2", w2)
    if p1 and p2:
        return distance_cal(w1, x1, y1, w2, x2, y2, dis)
    else:
        print("Has a photo without person")
        return 0


if __name__ == "__main__":
    yolov4, class_names, colors = load_network("/home/air/darknetab/cfg/yolov4-obj.cfg",
                                               "/home/air/darknetab/data/obj.data",
                                               "/home/air/darknetab/backup/yolov4-obj_final.weights")
    predict_dis(yolov4, class_names, colors, 1)
