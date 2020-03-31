#!/usr/bin/env python2
import struct
import cv2
import numpy as np
import socket
import json
import sys
from time import sleep
import ipdb
import cPickle as pickle

def encode(data):
    return json.dumps(data)

def decode(data):
    return json.loads(data)

HOST = '192.168.170.209'  # The server's hostname or IP address
PORT = 65432        # The port used by the server

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((HOST,PORT))
data = np.asarray([[1,2],[3,4]])
serialized_data = pickle.dumps(data,protocol=2)

client.send(serialized_data)
from_server = client.recv(4096)
from_server = pickle.loads(from_server)
client.close()

print(from_server)