import socket, time
print("START", flush=True)
time.sleep(2)
for host, port, label in [('8.8.8.8',53,'GoogleTCP53'),('1.1.1.1',443,'CFHTTPS'),('142.250.80.4',443,'GoogleHTTPS'),('192.168.64.1',80,'Gateway')]:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5); s.connect((host,port)); s.close()
        print(f"OK {label}", flush=True)
    except ConnectionRefusedError:
        print(f"REFUSED {label}", flush=True)
    except socket.timeout:
        print(f"TIMEOUT {label}", flush=True)
    except Exception as e:
        print(f"FAIL {label}", flush=True)
print("DONE", flush=True)
