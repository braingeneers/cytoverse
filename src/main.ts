import { createApp } from 'vue'
import { createVuetify } from 'vuetify'
import * as components from 'vuetify/components'
import * as directives from 'vuetify/directives'
import 'vuetify/styles'
import '@mdi/font/css/materialdesignicons.css'
import './index.css'
import App from './App.vue'

const vuetify = createVuetify({
  components,
  directives,
  theme: {
    defaultTheme: 'dark',
    themes: {
      dark: {
        dark: true,
        colors: {
          background: '#121212',
          surface: '#1e1e1e',
          primary: '#1976d2',
          secondary: '#424242',
          error: '#cf6679',
          info: '#2196f3',
          success: '#4caf50',
          warning: '#fb8c00',
        }
      }
    }
  }
})

createApp(App).use(vuetify).mount('#root')