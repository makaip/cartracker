export default defineAppConfig({
  ui: {
    colorMode: true,
    colors: {
      primary: 'sky',
      neutral: 'zinc'
    }
  },
  backend: {
    apiUrl: 'http://localhost:8765',
    wsUrl: 'ws://localhost:8765/ws',
    wsReconnectTimeout: 3000
  }
})
