import socket, time, ssl

print("START", flush=True)
time.sleep(2)

targets = [
    ("8.8.8.8", 443, False),
    ("1.1.1.1", 443, False),
    ("142.250.80.4", 443, False),  # google.com
]

for host, port, use_ssl in targets:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5)
        s.connect((host, port))
        print(f"TCP {host}:{port} CONNECTED", flush=True)
        if use_ssl:
            ctx = ssl.create_default_context()
            ss = ctx.wrap_socket(s, server_hostname=host)
            ss.settimeout(5)
            ss.do_handshake()
            print(f"TLS {host}:{port} OK", flush=True)
            ss.close()
        else:
            s.close()
    except ConnectionRefusedError:
        print(f"TCP {host}:{port} REFUSED", flush=True)
    except socket.timeout:
        print(f"TCP {host}:{port} TIMEOUT", flush=True)
    except Exception as e:
        print(f"TCP {host}:{port} {type(e).__name__}:{e}", flush=True)

print("DONE", flush=True)
