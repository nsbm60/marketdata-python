# zmq_subscriber.py
import zmq

ctx = zmq.Context()
sock = ctx.socket(zmq.SUB)
sock.connect("tcp://localhost:6006")
sock.setsockopt_string(zmq.SUBSCRIBE, "md.equity.quote.")

print("Listening...")
while True:
    print(sock.recv_string())
