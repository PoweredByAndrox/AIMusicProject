import './bootstrapFrontendRuntimeVersion'
import './bootstrapFrontendLogs'
import { createApp } from 'vue'
import { createPinia } from 'pinia'
import piniaPluginPersistedstate from 'pinia-plugin-persistedstate'
import Vue3Toastify, { type ToastContainerOptions } from 'vue3-toastify'
import 'vue3-toastify/dist/index.css'
import router from './router'
import App from './App.vue'
import '../css/app.css'
import { useAuthStore } from '@/stores/authStore'

const app = createApp(App)

const pinia = createPinia()
pinia.use(piniaPluginPersistedstate)

app.use(pinia)
app.use(router)
app.use(Vue3Toastify, {
    autoClose: 3000,
    position: 'top-right',
    theme: 'colored',
} as ToastContainerOptions)

const authStore = useAuthStore(pinia)
void authStore.initialize()

app.mount('#app')
