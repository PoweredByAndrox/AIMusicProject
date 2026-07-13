import socket, sys
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("192.168.64.1", 9999))
s.listen(1)
s.settimeout(15)
try:
    c, addr = s.accept()
    data = c.recv(256)
    print(f"GOT:{addr} DATA:{data!r}", flush=True)
    c.send(b"ACK\n")
    c.close()
except socket.timeout:
    print("TIMEOUT - no connection from container", flush=True)
s.close()
