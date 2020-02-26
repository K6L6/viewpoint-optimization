#!/usr/bin/env python3
import json
import socket
import select
import struct
import threading

import os
import sys
import math
import argparse
import numpy as np
import cupy as cp
import chainer
from chainer.backends import cuda

import gqn
from gqn.preprocessing import make_uint8, preprocess_images
from model_chain2 import Model
from hyperparams import HyperParameters
from functions import compute_yaw_and_pitch

server_HOST = '0.0.0.0' 
server_PORT = 65432

parser = argparse.ArgumentParser()
parser.add_argument("--gpu-device", type=int, default=0)
parser.add_argument("--snapshot-directory", "-snapshot", type=str, required=True)
# parser.add_argument("--dataset-directory", type=str, required=True)
# parser.add_argument("--figure-directory", type=str, required=True)
args = parser.parse_args()

def encode(data):
    return json.dumps(data)

def decode(data):
    return json.loads(data)

def compute_camera_angle_at_frame(t,total_frames):
    print("at frame: "+str(t)+" total frames: "+str(total_frames))
    return t*2*np.pi/total_frames

class SocketServer(threading.Thread):
    MAX_WAITING_CONNECTIONS = 5
    RECV_BUFFER = 4096
    RECV_MSG_LEN = 4
    def __init__(self, host, port, gpu, snapshot_dir, snapshot_file):
        threading.Thread.__init__(self)
        self.host = host
        self.port = port
        self.connections = []
        self.running = True

        my_gpu = args.gpu_device
        if my_gpu < 0:
            self.xp=np
        else:
            cuda.get_device(args.gpu_device).use()
            self.xp=cp
        hyperparams = HyperParameters()
        assert hyperparams.load(args.snapshot_directory)

        model = Model(hyperparams)
        chainer.serializers.load_hdf5(args.snapshot_file, model)
        if my_gpu > -1:
            model.to_gpu()
        
    def _bind_socket(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(self.MAX_WAITING_CONNECTIONS)
        self.connections.append(self.server_socket)
        
    @classmethod
    def to_gqn(data,model):
        
        observed_viewpoint, observed_image, offset = data
        observed_image = observed_image.transpose((0,1,4,2,3)).astype(np.float32)
        observed_image = preprocess_images(observed_image)

        # create representation and generate uncertainty map of environment [1000 viewpoints?]
        total_frames = 100
        representation = model.compute_observation_representation(observed_image, observed_viewpoint)

        # get predictions
        highest_var = 0.0
        no_of_samples = 100

        for i in range(0,total_frames):
            horizontal_angle_rad = compute_camera_angle_at_frame(i, total_frames)
            
            query_viewpoints = rotate_query_viewpoint(
                        horizontal_angle_rad, camera_distance, camera_position_z)
            
            generated_images = xp.squeeze(xp.array(model.generate_images(query_viewpoints,
                                                                representation,no_of_samples)))
            var_image = xp.var(generated_images,axis=0)
            var_image = chainer.backends.cuda.to_cpu(var_image)
            # grayscale
            r,g,b = var_image
            gray_var_image = 0.2989*r+0.5870*g+0.1140*b
            current_var = np.mean(gray_var_image)
            
            if current_var>highest_var:
                highest_var = current_var
                highest_var_vp = query_viewpoints[0]

        # return next viewpoint and unit vector of end effector based on highest uncertainty found in the uncertainty map
        _x, _y, _z, _, _, _, _ = highest_var_vp
        _yaw, _pitch = compute_yaw_and_pitch([_x, _y, _z])
        pose_x, pose_y, pose_z = GQN_VP2gazeboPose([_x,_y,_z],offset) # convert _x, _y, _z to end effector values??

        return next_viewpoint

    def _send(cls, sock, msg):
        # Append message with length of message
        msg = struct.pack(">I", len(msg)) + msg.encode("utf-8")
        sock.send(msg)
        
    def _receive(self, sock):
        data = None
        total_len = 0
        while total_len < self.RECV_MSG_LEN:
            msg_len = sock.recv(self.RECV_MSG_LEN)
            total_len = total_len + len(msg_len)
            
        # If the message has the length prefix
        if msg_len:
            data = ""
            msg_len = struct.unpack(">I", msg_len)[0]
            total_data_len = 0
            while total_data_len < msg_len:
                chunk = sock.recv(self.RECV_BUFFER)
                
                if not chunk:
                    data = None
                    break
                else:
                    data = data + chunk.decode("utf-8")
                    total_data_len = total_data_len + len(chunk)
        
        gqn = threading.Thread(target=self.to_gqn(),args=(data,model))
        next_vp = gqn.start()
        gqn.join()
        
        return data
    
    def _broadcast(self, client_socket, client_message):
        print("Starting broadcast...")
        for sock in self.connections:
            not_server = (sock != self.server_socket)
            not_sending_client = (sock != client_socket)
            if not_server and not_sending_client:
                try:
                    print("Broadcasting: %s" % client_message)
                    self._send(sock, client_message)
                except socket.error:
                    # Client no longer replying
                    print("Closing a socket...")
                    sock.close()
                    self.connections.remove(sock)

    def _run(self):
        print("Starting socket server (%s, %s)..." % (self.host, self.port))
        while self.running:
            try:
                # Timeout every 60 seconds
                selection = select.select(self.connections, [], [], 5)
                read_sock = selection[0]
            except select.error:
                print("Error!!!")
            else:
                for sock in read_sock:
                    # New connection
                    if sock == self.server_socket:
                        try:
                            accept_sock = self.server_socket.accept()
                            client_socket, client_address = accept_sock
                        except socket.error:
                            print("Other error!")
                            break
                        else:
                            self.connections.append(client_socket)
                            print("Client (%s, %s) is online" % client_address)
                            
                            self._broadcast(client_socket, encode({
                                "name": "connected",
                                "data": client_address
                            }))
                    # Existing client connection
                    else:
                        try:
                            data = self._receive(sock)
                            if data:
                                print(decode(data))
                                self._send(sock, data)
                        except socket.error:
                            # Client is no longer replying
                            self._broadcast(sock, encode({
                                "name": "disconnected",
                                "data": client_address
                            }))
                            print("Client (%s, %s) is offline" % client_address)
                            sock.close()
                            self.connections.remove(sock)
                            continue
        # Clear the socket connection
        self.stop()
    
    def run(self):
        self._bind_socket()
        self._run()
        
    def stop(self):
        self.running = False
        self.server_socket.close()
    
if __name__ == "__main__":
    socket_server = SocketServer(server_HOST, server_PORT)
    socket_server.start()