import socket, time, struct

print("START", flush=True)
time.sleep(2)

# Test UDP connectivity
# DNS query to 8.8.8.8:53
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.settimeout(5)
dns_query = bytes([
    0x12, 0x34,  # Transaction ID
    0x01, 0x00,  # Flags: standard query, recursion desired
    0x00, 0x01,  # Questions: 1
    0x00, 0x00,  # Answer RRs: 0
    0x00, 0x00,  # Authority RRs: 0
    0x00, 0x00,  # Additional RRs: 0
    # Query for google.com (A record)
    0x06, ord('g'), ord('o'), ord('o'), ord('g'), ord('l'), ord('e'),
    0x03, ord('c'), ord('o'), ord('m'),
    0x00,        # End of domain name
    0x00, 0x01,  # Type: A
    0x00, 0x01,  # Class: IN
])
try:
    s.sendto(dns_query, ("8.8.8.8", 53))
    data, addr = s.recvfrom(1024)
    print(f"UDP DNS 8.8.8.8:53 RESPONSE from {addr} len={len(data)}", flush=True)
except socket.timeout:
    print("UDP DNS 8.8.8.8:53 TIMEOUT", flush=True)
except Exception as e:
    print(f"UDP DNS 8.8.8.8:53 {type(e).__name__}:{e}", flush=True)
s.close()

# UDP to 1.1.1.1:53
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.settimeout(5)
try:
    s.sendto(dns_query, ("1.1.1.1", 53))
    data, addr = s.recvfrom(1024)
    print(f"UDP DNS 1.1.1.1:53 RESPONSE from {addr} len={len(data)}", flush=True)
except socket.timeout:
    print("UDP DNS 1.1.1.1:53 TIMEOUT", flush=True)
except Exception as e:
    print(f"UDP DNS 1.1.1.1:53 {type(e).__name__}:{e}", flush=True)
s.close()

# TCP port 22 to google.com
for host, port in [("142.250.80.4", 22), ("142.250.80.4", 443)]:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5)
        s.connect((host, port))
        s.close()
        print(f"TCP {host}:{port} OK", flush=True)
    except ConnectionRefusedError:
        print(f"TCP {host}:{port} REFUSED", flush=True)
    except socket.timeout:
        print(f"TCP {host}:{port} TIMEOUT", flush=True)
    except Exception as e:
        print(f"TCP {host}:{port} {type(e).__name__}", flush=True)

print("DONE", flush=True)
