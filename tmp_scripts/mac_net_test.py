#!/usr/bin/env python3
import urllib.request
try:
    r = urllib.request.urlopen('https://pypi.org/simple/', timeout=10)
    print(f"PYPI_OK:{r.status}")
except Exception as e:
    print(f"PYPI_FAIL:{e}")

try:
    r = urllib.request.urlopen('https://google.com', timeout=10)
    print(f"WEB_OK:{r.status}")
except Exception as e:
    print(f"WEB_FAIL:{e}")
