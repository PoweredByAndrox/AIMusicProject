import urllib.request, time
print("START", flush=True)
time.sleep(2)
print("SLEPT", flush=True)
try:
    r = urllib.request.urlopen("https://pypi.org/simple/", timeout=5)
    print(f"OK:{r.status}", flush=True)
except Exception as e:
    print(f"FAIL:{type(e).__name__}:{e}", flush=True)
print("DONE", flush=True)
