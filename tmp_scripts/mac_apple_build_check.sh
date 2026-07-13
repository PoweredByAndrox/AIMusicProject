#!/bin/bash
set -e
rm -rf /tmp/apple_build 2>/dev/null || true
mkdir -p /tmp/apple_build

cat > /tmp/apple_build/test_net.py << 'PYEOF'
import urllib.request
r = urllib.request.urlopen('https://pypi.org/simple/', timeout=10)
print('PYPI_OK:', r.status)
try:
    r2 = urllib.request.urlopen('https://google.com', timeout=10)
    print('WEB_OK:', r2.status)
except Exception as e:
    print('WEB_FAIL:', e)
    # debian might not work inside container
    print('DEBIAN_NETWORK: unknown (this usually fails in Apple container VMs)')
PYEOF

echo "#!/bin/bash" > /tmp/apple_build/Dockerfile.test
echo "set -e" >> /tmp/apple_build/Dockerfile.test
echo "python3 /app/test_net.py" >> /tmp/apple_build/Dockerfile.test
chmod +x /tmp/apple_build/Dockerfile.test

cat > /tmp/apple_build/Dockerfile << 'DOCKERFILE'
FROM python:3.11-slim
WORKDIR /app
COPY test_net.py .
RUN python3 test_net.py
DOCKERFILE

cat > /tmp/apple_build/requirements.txt << 'REQS'
fastapi==0.115.0
uvicorn[standard]==0.30.6
pydantic==2.9.2
packaging>=24.1
pytest==8.3.3
REQS

echo "FILES_READY"
ls -la /tmp/apple_build/
