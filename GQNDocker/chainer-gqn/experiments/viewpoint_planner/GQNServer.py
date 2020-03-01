#!/usr/bin/env python3
import json
import socket
import select
import struct
import threading
import ipdb
from six.moves import queue

import os
import sys
import math
import argparse
import numpy as np
import cupy as cp
import chainer
from chainer.backends import cuda

sys.path.append("./../../")
import gqn
from gqn.preprocessing import make_uint8, preprocess_images
from model_chain import Model
from hyperparams import HyperParameters
from functions import compute_yaw_and_pitch

def encode(data):
    return json.dumps(data)

def decode(data):
    return json.loads(data)

def compute_camera_angle_at_frame(t,total_frames):
    print("at frame: "+str(t)+" total frames: "+str(total_frames))
    return t*2*np.pi/total_frames

def rotate_query_viewpoint(horizontal_angle_rad, camera_distance,
                               camera_position_y,xp):
    camera_position = np.array([
        camera_distance * math.sin(horizontal_angle_rad),   # x
        camera_position_y,
        camera_distance * math.cos(horizontal_angle_rad),  # z
    ])
    center = np.array((0, camera_position_y, 0)) 
    camera_direction = camera_position - center
    yaw, pitch = compute_yaw_and_pitch(camera_direction)
        
    query_viewpoints = xp.array(
        (
            camera_position[0],
            camera_position[1],
            camera_position[2],
            math.cos(yaw),
            math.sin(yaw),
            math.cos(pitch),
            math.sin(pitch),
        ),
        dtype=np.float32,
    )
    query_viewpoints = xp.broadcast_to(query_viewpoints,
                                        (1, ) + query_viewpoints.shape)

    return query_viewpoints

def gqn_process():
    # load model
    my_gpu = args.gpu_device
    if my_gpu < 0:
        xp=np
    else:
        cuda.get_device(args.gpu_device).use()
        xp=cp
    hyperparams = HyperParameters()
    assert hyperparams.load(args.snapshot_directory)

    model = Model(hyperparams)
    chainer.serializers.load_hdf5(args.snapshot_file, model)
    if my_gpu > -1:
        model.to_gpu()
    chainer.print_runtime_info()

    observed_viewpoint, observed_image, offset = data_recv.get()
    observed_viewpoint = np.expand_dims(np.expand_dims(np.asarray(observed_viewpoint).astype(np.float32),axis=0),axis=0)
    observed_image = np.expand_dims(np.expand_dims(np.asarray(observed_image).astype(np.float32),axis=0),axis=0)
    offset = np.asarray(offset)

    camera_distance = np.mean(np.linalg.norm(observed_viewpoint[:,:,:3],axis=2))
    camera_position_z = np.mean(observed_viewpoint[:,:,1])
    # ipdb.set_trace()
    observed_image = observed_image.transpose((0,1,4,2,3)).astype(np.float32)
    observed_image = preprocess_images(observed_image)

    # create representation and generate uncertainty map of environment [1000 viewpoints?]
    total_frames = 100
    representation = model.compute_observation_representation(observed_image, observed_viewpoint)

    # get predictions
    highest_var = 0.0
    no_of_samples = 100
    try:
        for i in range(0,total_frames):
            horizontal_angle_rad = compute_camera_angle_at_frame(i, total_frames)
            
            query_viewpoints = rotate_query_viewpoint(
                        horizontal_angle_rad, camera_distance, camera_position_z,xp)
            
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
    except KeyboardInterrupt:
        print('interrupt')

    # return next viewpoint and unit vector of end effector based on highest uncertainty found in the uncertainty map
    _x, _y, _z, _, _, _, _ = highest_var_vp
    _yaw, _pitch = compute_yaw_and_pitch([_x, _y, _z])
    next_viewpoint = [_x, _y, _z, _yaw, _pitch]
    data_send.put(next_viewpoint)

class SocketServer(threading.Thread):
    MAX_WAITING_CONNECTIONS = 5
    RECV_BUFFER = 131072 # 2^17
    RECV_MSG_LEN = 4
    def __init__(self, host, port):
        threading.Thread.__init__(self)
        self.host = host
        self.port = port
        self.connections = []
        self.running = True
        
    def _bind_socket(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(self.MAX_WAITING_CONNECTIONS)
        self.connections.append(self.server_socket)
        
    @classmethod
    def _send(cls, sock, data):
        data=encode(data)
        sock.send(data)
        
    def _receive(self, sock):
        data = sock.recvmsg(self.RECV_BUFFER)
        # ipdb.set_trace()
        data = decode(data[0])
        
        data_recv.put(data)
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
                            if not (data_send.empty()):
                                send_data = data_send.get()
                                self._send(sock, send_data)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--snapshot-directory", "-snapdir", type=str, required=True)
    parser.add_argument("--snapshot-file", "-snapfile", type=str, required=True)
    # parser.add_argument("--dataset-directory", type=str, required=True)
    # parser.add_argument("--figure-directory", type=str, required=True)
    args = parser.parse_args()

    server_HOST = '0.0.0.0' 
    server_PORT = 65432

    data_recv = queue.Queue(maxsize=1)
    data_send = queue.Queue(maxsize=1)
    socket_server = SocketServer(server_HOST, server_PORT)

    gqn_work = threading.Thread(target=gqn_process, daemon=True)
    gqn_work.start()

    socket_server.start()
    gqn_work.join()
    