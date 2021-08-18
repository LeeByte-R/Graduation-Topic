import socket
import threading
import time
import numpy as np
import libh264decoder


"""Python3"""
class Tello:
    def __init__(self, local_ip, local_port, imperial=False, command_timeout=5,
                 tello_ip='192.168.10.1', tello_port=8889):
        self.abort_flag = False
        self.decoder = libh264decoder.H264Decoder()
        self.command_timeout = command_timeout
        self.imperial = imperial
        self.response = None  
        self.frame = None  # numpy array BGR -- current camera output frame
        self.is_freeze = False  # freeze current camera output
        self.last_frame = None
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # socket for sending cmd
        self.socket_video = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # socket for receiving video stream
        self.tello_address = (tello_ip, tello_port)
        self.local_video_port = 11111  # port for receiving video stream
        self.last_height = 0
        self.socket.bind((local_ip, local_port))

        # thread for receiving cmd ack
        self.receive_thread = threading.Thread(target=self._receive_thread)
        self.receive_thread.daemon = True
        self.receive_thread.start()

        # to receive video -- send cmd: command, streamon
        self.send_command('command')
        # self.socket.sendto(b'command', self.tello_address)
        # print('sent: command')
        self.socket.sendto(b'streamon', self.tello_address)
        # self.send_command('streamon')
        # print('sent: streamon')

        self.socket_video.bind(("0.0.0.0", self.local_video_port))

        # thread for receiving video
        self.receive_video_thread = threading.Thread(target=self._receive_video_thread)
        self.receive_video_thread.daemon = True
        self.receive_video_thread.start()

    def __del__(self):
        self.socket.close()
        self.socket_video.close()
    
    def read(self):
        if self.is_freeze:
            return self.last_frame
        else:
            return self.frame

    def video_freeze(self, is_freeze=True):
        self.is_freeze = is_freeze
        if is_freeze:
            self.last_frame = self.frame

    def _receive_thread(self):
        while True:
            try:
                self.response, ip = self.socket.recvfrom(3000)
                # print(self.response)
            except socket.error as exc:
                print(("Caught exception socket.error : %s" % exc))

    def _receive_video_thread(self):
        packet_data = b''
        while True:
            try:
                res_string, ip = self.socket_video.recvfrom(2048)
                packet_data += res_string
                # end of frame
                if len(res_string) != 1460:
                    for frame in self._h264_decod(packet_data):
                        self.frame = frame
                    packet_data = b''
                        
            except socket.error as exc:
                print(("Caught exception socket.error : %s" % exc))

    def _h264_decod(self, packet_data):
        res_frame_list = []
        frames = self.decoder.decode(packet_data)
                 
        for framedata in frames:
            (frame, w, h, ls) = framedata
            if frame is not None:
                # print ('frame size %i bytes, w %i, h %i, linesize %i' % (len(frame), w, h, ls))
                frame = np.frombuffer(frame, dtype = np.ubyte, count = len(frame))
                frame = (frame.reshape((h, ls//3, 3)))
                frame = frame[:, :w, :]
                res_frame_list.append(frame)

        return res_frame_list

    def send_command(self, command):
        print(">> send cmd: {}".format(command))
        while True:
            self.abort_flag = False
            timer = threading.Timer(self.command_timeout, self.set_abort_flag)
            self.socket.sendto(command.encode('utf-8'), self.tello_address)
            timer.start()
            while self.response is None:
                if self.abort_flag is True:
                    break
            timer.cancel()

            if self.response is None:
                response = 'none_response'
            else:
                response = self.response.decode('latin-1')

            if response == "ok":
                print(response)
                break
            time.sleep(0.1)
            print(">>resend cmd " + command)

        self.response = None
        return response

    def takeoff(self):
        return self.send_command('takeoff')

    def set_speed(self, speed):
        speed = float(speed)

        if self.imperial is True:
            speed = int(round(speed * 44.704))
        else:
            speed = int(round(speed * 27.7778))

        return self.send_command('speed %s' % speed)

    def rotate_cw(self, degrees):
        return self.send_command('cw %s' % degrees)

    def rotate_ccw(self, degrees):
        return self.send_command('ccw %s' % degrees)

    def get_response(self):
        response = self.response
        return response

    def get_height(self):
        height = self.send_command('height?')
        height = str(height)
        height = list(filter(str.isdigit, height))
        try:
            height = int(height)
            self.last_height = height
        except:
            height = self.last_height
            pass
        return height

    def get_battery(self):
        battery = self.send_command('battery?')
        try:
            battery = int(battery)
        except:
            pass

        return battery

    def get_flight_time(self):
        flight_time = self.send_command('time?')

        try:
            flight_time = int(flight_time)
        except:
            pass

        return flight_time

    def get_speed(self):
        speed = self.send_command('speed?')
        try:
            speed = float(speed)

            if self.imperial is True:
                speed = round((speed / 44.704), 1)
            else:
                speed = round((speed / 27.7778), 1)
        except:
            pass

        return speed

    def land(self):
        return self.send_command('land')

    def move(self, direction, distance):
        distance = float(distance)

        if self.imperial is True:
            distance = int(round(distance * 30.48))
        else:
            distance = int(round(distance * 100))

        return self.send_command('%s %s' % (direction, distance))

    def move_backward(self, distance):
        return self.move('back', distance)

    def move_down(self, distance):
        return self.move('down', distance)

    def move_forward(self, distance):
        return self.move('forward', distance)

    def move_left(self, distance):
        return self.move('left', distance)

    def move_right(self, distance):
        return self.move('right', distance)

    def move_up(self, distance):
        return self.move('up', distance)

    def set_abort_flag(self):
        self.abort_flag = True
