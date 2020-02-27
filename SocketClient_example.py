#!/usr/bin/env python3
import json
import struct
import socket

# type: obj -> str
def encode(data):
    return json.dumps(data)

# type: str -> obj
def decode(data):
    return json.loads(data)

class SocketClient():
    def __init__(host, port, data, recv_msg_len, recv_buffer):
        self.host = host
        self.port = port
        self.recv_msg_len = recv_msg_len
        self.recv_buffer = recv_buffer

    def run(self):
        s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)

        # connect to server on local computer
        s.connect((self.host,self.port))

        # message you send to server
        message = encode({data})
        # message = encode({
        #     "name": "test-message",
        #     "message": "sending message to socket client"
        # })
        
        while True:
            # message sent to server
            msg = struct.pack(">I", len(message)) + message.encode("utf-8")
            s.send(msg)

            # messaga received from server
            data = None
            total_len = 0
            while total_len < self.recv_msg_len:
                msg_len = s.recv(self.recv_msg_len)
                total_len = total_len + len(msg_len)
                
            # If the message has the length prefix
            if msg_len:
                data = ""
                msg_len = struct.unpack(">I", msg_len)[0]
                total_data_len = 0
                while total_data_len < msg_len:
                    chunk = s.recv(self.recv_buffer)
                    
                    if not chunk:
                        data = None
                        break
                    else:
                        data = data + chunk.decode("utf-8")
                        total_data_len = total_data_len + len(chunk)
                        
            # print the received message
            # here it would be a reverse of sent message
            print("Received from the server :", decode(data).get("message", ""))

            # ask the client whether he wants to continue
            print("\nDo you want to continue(y/n) :")
            ans = raw_input()
            if ans == "y":
                continue
            else:
                break
                
        # close the connection
        s.close()
        
if __name__ == "__main__":
    main("192.168.170.209", 65432, 4, 1024)