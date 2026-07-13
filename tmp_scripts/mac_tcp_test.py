import socket, time

print("START", flush=True)
time.sleep(2)

# Try raw TCP connection to 8.8.8.8:53 (DNS) or 1.1.1.1:80 (HTTP)
for host, port in [("8.8.8.8", 53), ("1.1.1.1", 80), ("192.168.64.1", 80)]:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(3)
        s.connect((host, port))
        print(f"TCP {host}:{port} OK", flush=True)
        s.close()
    except Exception as e:
        print(f"TCP {host}:{port} FAIL {type(e).__name__}", flush=True)

print("DONE", flush=True)
