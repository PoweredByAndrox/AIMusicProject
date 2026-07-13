import time, socket, urllib.request

time.sleep(5)
print("Hostname:", socket.gethostname())

try:
    addrs = socket.getaddrinfo("pypi.org", 443)
    print("DNS pypi.org:", addrs[0][4])
except Exception as e:
    print("DNS fail:", e)

try:
    r = urllib.request.urlopen("https://pypi.org/simple/", timeout=15)
    print("HTTP:", r.status)
except Exception as e:
    print("HTTP fail:", e)

try:
    r = urllib.request.urlopen("http://8.8.8.8", timeout=10)
    print("RAW-HTTP:", r.status)
except Exception as e:
    print("RAW-HTTP fail:", e)
