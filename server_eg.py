#!/usr/bin/env python3
import json
import logging
import socket
import select
import struct
import numpy as np
from time import sleep
import threading
import ipdb
from six.moves import queue
import pickle

def encode(data):
    print(data)
    return json.dumps(data)

def decode(data):
    return json.loads(data)

server_HOST = '0.0.0.0' 
server_PORT = 65432

data_recv = queue.Queue(maxsize=1)
data_send = queue.Queue(maxsize=1)

serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
serv.bind((server_HOST, server_PORT))
serv.listen(5)

while True:
    conn, addr = serv.accept()
    
    while True:
        data = conn.recv(4096)
        if not data: break
        data = pickle.loads(data,encoding='bytes')
        out = data + np.array([[1,0],[0,1]])
        conn.send(pickle.dumps(out, protocol=2))
        
        print(type(data))
conn.close()
print('client disconnected')