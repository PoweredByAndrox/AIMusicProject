# Cloudflare Quick Tunnel для localhost:8000

Пробрасывает `http://localhost:8000` наружу через Cloudflare. Без регистрации, без домена.

## Запуск

1. Установи [cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/)
2. Убедись что сайт работает на `http://localhost:8000`
3. Запусти `start_tunnel.bat`
4. В консоли появится URL вида `https://random-words.trycloudflare.com`
5. Открой `https://random-words.trycloudflare.com/radio/test`

URL временный — меняется при каждом перезапуске.
