#!/bin/sh
set -eu

ensure_dir() {
    mkdir -p "$1"
}

current_lock_hash() {
    if [ -f /var/www/html/composer.lock ] || [ -f /var/www/html/composer.json ]; then
        sha1sum /var/www/html/composer.lock /var/www/html/composer.json 2>/dev/null | sha1sum | awk '{print $1}'
    else
        printf ''
    fi
}

ensure_vendor() {
    ensure_dir /var/log/app

    lock_hash="$(current_lock_hash)"
    installed_hash=""

    if [ -f /var/www/html/vendor/.composer-lock-hash ]; then
        installed_hash="$(cat /var/www/html/vendor/.composer-lock-hash 2>/dev/null || true)"
    fi

    if [ ! -f /var/www/html/vendor/autoload.php ] || [ "$lock_hash" != "$installed_hash" ]; then
        echo "[app] running composer install" >> /var/log/app/composer-install.log
        composer install --no-interaction --prefer-dist >> /var/log/app/composer-install.log 2>&1
        ensure_dir /var/www/html/vendor
        printf '%s' "$lock_hash" > /var/www/html/vendor/.composer-lock-hash
    else
        echo "[app] reusing cached vendor" >> /var/log/app/composer-install.log
    fi
}

fix_tree() {
    if [ -e "$1" ]; then
        chown -R www-data:www-data "$1"
        find "$1" -type d -exec chmod 775 {} \;
        find "$1" -type f -exec chmod 664 {} \;
    fi
}

ensure_dir /var/www/html/storage/logs
ensure_dir /var/www/html/storage/app/audio
ensure_dir /var/www/html/storage/framework/cache
ensure_dir /var/www/html/storage/framework/sessions
ensure_dir /var/www/html/storage/framework/views
ensure_dir /var/www/html/bootstrap/cache
ensure_vendor

fix_tree /var/www/html/storage/logs
fix_tree /var/www/html/storage/framework
fix_tree /var/www/html/bootstrap/cache

# The audio import directory lives on a bind mount under Windows/WSL.
# Ownership normalization can be ineffective there, so force world-writable
# on the directory itself to keep youtube queue workers able to create files.
chmod 0777 /var/www/html/storage/app/audio || true

exec /usr/bin/supervisord -c /etc/supervisor.d/supervisord.ini
