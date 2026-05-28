<?php

namespace App\Support;

final class AppStoragePath
{
    public static function data(string $path = ''): string
    {
        $configured = env('STORAGE_PATH');
        $root = is_string($configured) && trim($configured) !== ''
            ? rtrim($configured, "/\\")
            : storage_path('app');

        if ($path === '') {
            return $root;
        }

        return $root . DIRECTORY_SEPARATOR . ltrim($path, "/\\");
    }

    public static function log(string $path = ''): string
    {
        if ($path === '') {
            return self::data('logs');
        }

        return self::data('logs' . DIRECTORY_SEPARATOR . ltrim($path, "/\\"));
    }
}
