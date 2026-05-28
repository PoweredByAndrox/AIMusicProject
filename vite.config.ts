import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import laravel from 'laravel-vite-plugin'
import { resolve } from 'path'

const frontendRuntimeVersion = process.env.FRONTEND_RUNTIME_VERSION ?? `${Date.now()}`

export default defineConfig({
    define: {
        __FRONTEND_RUNTIME_VERSION__: JSON.stringify(frontendRuntimeVersion),
    },
    plugins: [
        laravel({
            input: ['resources/css/app.css', 'resources/js/app.ts', 'resources/js/public-radio.ts'],
            refresh: true,
        }),
        vue({
            template: {
                transformAssetUrls: {
                    base: null,
                    includeAbsolute: false,
                },
            },
        }),
    ],
    resolve: {
        alias: {
            '@': resolve(__dirname, 'resources/js'),
        },
    },
    build: {
        // Хеширование всех файлов для cache busting
        rollupOptions: {
            output: {
                // JS файлы с хешем содержимого
                entryFileNames: 'assets/[name]-[hash].js',
                chunkFileNames: 'assets/[name]-[hash].js',
                // CSS и другие ассеты с хешем
                assetFileNames: 'assets/[name]-[hash].[ext]',
            },
        },
        // Генерировать manifest для Laravel
        manifest: true,
    },
    server: {
        host: '0.0.0.0',
        port: 5173,
        strictPort: true,
        hmr: {
            host: 'localhost',
            port: 5173,
        },
        proxy: {
            '/api': {
                target: 'http://localhost:8000',
                changeOrigin: true,
            },
        },
    },
})
