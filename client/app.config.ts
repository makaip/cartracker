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
    // number of server nodes (server/config.yaml job.nodes); node i is on port 8765 + i
    nodes: 2,
    wsReconnectTimeout: 3000
  }
})
