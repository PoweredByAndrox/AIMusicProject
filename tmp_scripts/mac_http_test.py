import socket, time, http.client

print("START", flush=True)
time.sleep(2)

# Test DNS vs raw IP
hosts_to_check = [
    ("example.com", 80, False),
    ("httpbin.org", 80, False),
    ("93.184.216.34", 80, False),    # example.com IP
]

for host, port, use_ssl in hosts_to_check:
    try:
        # DNS test
        ip = socket.gethostbyname(host) if not host.replace('.','').isdigit() else host
        print(f"DNS {host} -> {ip}", flush=True)

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5)
        s.connect((ip, port))
        print(f"TCP {ip}:{port} CONNECTED", flush=True)

        # Send HTTP request
        msg = f"GET / HTTP/1.0\r\nHost: {host}\r\n\r\n"
        s.send(msg.encode())
        resp = s.recv(1024)
        print(f"HTTP {host} RESPONSE: {resp[:100]}", flush=True)
        s.close()
    except Exception as e:
        print(f"FAIL {host}:{port} {type(e).__name__}:{e}", flush=True)

print("DONE", flush=True)
