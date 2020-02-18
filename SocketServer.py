#!/usr/bin/env python3

import socket
import select
import struct
import threading

server_HOST = '0.0.0.0' 
server_PORT = 65432

class SocketServer(threading.Thread):
    #something for socket server