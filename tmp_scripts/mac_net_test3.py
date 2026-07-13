import socket, time

print("START", flush=True)
time.sleep(2)

# Test various ports to understand connectivity
tests = [
    ("8.8.8.8", 53, "Google DNS TCP"),
    ("8.8.8.8", 443, "Google DNS HTTPS"),
    ("1.1.1.1", 53, "Cloudflare DNS TCP"),
    ("1.1.1.1", 80, "Cloudflare HTTP"),
    ("192.168.64.1", 80, "Gateway HTTP"),
    ("192.168.64.1", 9999, "Gateway custom"),
]

for host, port, label in tests:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5)
        s.connect((host, port))
        s.close()
        print(f"OK {label} {host}:{port}", flush=True)
    except ConnectionRefusedError:
        print(f"REFUSED {label} {host}:{port}", flush=True)
    except socket.timeout:
        print(f"TIMEOUT {label} {host}:{port}", flush=True)
    except OSError as e:
        print(f"OSERR {label} {host}:{port} {e}", flush=True)
    except Exception as e:
        print(f"FAIL {label} {host}:{port} {type(e).__name__}", flush=True)

print("DONE", flush=True)
