# Laravel Application Dockerfile
# PHP 8.2+ with nginx for AI Music Recommendations
# Requirements: 13.1

FROM php:8.2-fpm-alpine

# Install system dependencies
RUN apk add --no-cache \
    nginx \
    supervisor \
    curl \
    git \
    unzip \
    libpng-dev \
    libjpeg-turbo-dev \
    freetype-dev \
    libzip-dev \
    postgresql-dev \
    icu-dev \
    oniguruma-dev \
    linux-headers \
    python3 \
    py3-pip \
    ffmpeg \
    nodejs \
    npm \
    $PHPIZE_DEPS

# Install yt-dlp for YouTube downloads
RUN pip3 install --break-system-packages yt-dlp \
    && yt-dlp --version

# Install PHP extensions
RUN docker-php-ext-configure gd --with-freetype --with-jpeg \
    && docker-php-ext-install -j$(nproc) \
        pdo \
        pdo_pgsql \
        pgsql \
        pcntl \
        bcmath \
        gd \
        zip \
        intl \
        mbstring \
        opcache

# Install Redis extension via PECL
RUN pecl install redis \
    && docker-php-ext-enable redis

# Install Composer
COPY --from=composer:2 /usr/bin/composer /usr/bin/composer

# Configure PHP
RUN mv "$PHP_INI_DIR/php.ini-production" "$PHP_INI_DIR/php.ini"

# PHP configuration for Laravel
RUN echo "upload_max_filesize = 100M" >> "$PHP_INI_DIR/conf.d/laravel.ini" \
    && echo "post_max_size = 100M" >> "$PHP_INI_DIR/conf.d/laravel.ini" \
    && echo "memory_limit = 256M" >> "$PHP_INI_DIR/conf.d/laravel.ini" \
    && echo "max_execution_time = 300" >> "$PHP_INI_DIR/conf.d/laravel.ini"

# OPcache configuration for production
RUN echo "opcache.enable=1" >> "$PHP_INI_DIR/conf.d/opcache.ini" \
    && echo "opcache.memory_consumption=128" >> "$PHP_INI_DIR/conf.d/opcache.ini" \
    && echo "opcache.interned_strings_buffer=8" >> "$PHP_INI_DIR/conf.d/opcache.ini" \
    && echo "opcache.max_accelerated_files=10000" >> "$PHP_INI_DIR/conf.d/opcache.ini" \
    && echo "opcache.validate_timestamps=0" >> "$PHP_INI_DIR/conf.d/opcache.ini"

# Configure PHP-FPM to use TCP port and a larger local worker pool
RUN sed -i 's/listen = 127.0.0.1:9000/listen = 9000/' /usr/local/etc/php-fpm.d/www.conf \
    && sed -i 's/^pm.max_children = 5/pm.max_children = 10/' /usr/local/etc/php-fpm.d/www.conf \
    && sed -i 's/^pm.start_servers = 2/pm.start_servers = 4/' /usr/local/etc/php-fpm.d/www.conf \
    && sed -i 's/^pm.min_spare_servers = 1/pm.min_spare_servers = 2/' /usr/local/etc/php-fpm.d/www.conf \
    && sed -i 's/^pm.max_spare_servers = 3/pm.max_spare_servers = 6/' /usr/local/etc/php-fpm.d/www.conf

# Create nginx directories
RUN mkdir -p /run/nginx

# Configure nginx - replace default config in http.d
COPY docker/app/nginx.conf /etc/nginx/http.d/default.conf

# Create supervisor configuration
RUN mkdir -p /etc/supervisor.d
COPY docker/app/supervisord.conf /etc/supervisor.d/supervisord.ini
COPY docker/app/start.sh /usr/local/bin/start-app
RUN chmod +x /usr/local/bin/start-app

# Set working directory
WORKDIR /var/www/html

# Create storage directories
RUN mkdir -p /var/www/html/storage/app/public \
    /var/www/html/storage/framework/cache \
    /var/www/html/storage/framework/sessions \
    /var/www/html/storage/framework/views \
    /var/www/html/storage/logs \
    /var/www/html/bootstrap/cache

# Set permissions
RUN chown -R www-data:www-data /var/www/html \
    && chmod -R 775 /var/www/html/storage \
    && chmod -R 775 /var/www/html/bootstrap/cache

# Expose port 80
EXPOSE 80

# Start app runtime and normalize writable Laravel paths on container boot
CMD ["/usr/local/bin/start-app"]
