import { computed, shallowRef, triggerRef, watch } from 'vue'
import { cameraService } from '../services/cameraService'
import type { CameraOption, CameraMap, CameraStatusMap, CameraDetectionMap } from '../types/camera'

interface Backend {
  apiUrl: string
  wsUrl: string
}

// one entry per server node; node i listens on the configured port + i
function resolveBackends(config: any): Backend[] {
  const apiUrl: string = config?.apiUrl ?? 'http://localhost:8765'
  const wsUrl: string = config?.wsUrl ?? 'ws://localhost:8765/ws'
  const nodes = Math.max(1, Number(config?.nodes) || 1)

  const withPortOffset = (url: string, offset: number) => {
    if (offset === 0) return url
    const parsed = new URL(url)
    parsed.port = String(Number(parsed.port) + offset)
    return parsed.href.replace(/\/$/, '')
  }

  return Array.from({ length: nodes }, (_, i) => ({
    apiUrl: withPortOffset(apiUrl, i),
    wsUrl: withPortOffset(wsUrl, i)
  }))
}

// per-node state, indexed like the backend list
const sockets: (WebSocket | null)[] = []
const camerasByNode: (CameraMap | null)[] = []
const statusByNode: CameraStatusMap[] = []
// camera id -> apiUrl of the node that serves it
const cameraBackend = shallowRef<Record<string, string>>({})
const connectedCount = shallowRef(0)
let wsShouldReconnect = false
let statusPollingInterval: NodeJS.Timeout | null = null

// The server sends hundreds of detection messages per second (one per camera per
// processed frame). Writing each one into deep reactive state re-rendered the whole
// page per message, so messages land in a plain object and the UI is notified at a
// bounded rate instead.
const MAP_FLUSH_MS = 500

// latest detections per camera; mutated in place, never made deeply reactive
const detectionStore: CameraDetectionMap = {}
// same object as detectionStore; triggered at most once per MAP_FLUSH_MS (map markers)
const cameraDetections = shallowRef<CameraDetectionMap>(detectionStore)
// bumped at most once per animation frame when the selected camera gets new detections
const selectedTick = shallowRef(0)

let mapFlushTimer: ReturnType<typeof setTimeout> | null = null
let selectedFramePending = false
let lastStatusJson = ''

function scheduleMapFlush() {
  if (mapFlushTimer) return
  mapFlushTimer = setTimeout(() => {
    mapFlushTimer = null
    triggerRef(cameraDetections)
  }, MAP_FLUSH_MS)
}

function scheduleSelectedFlush() {
  if (selectedFramePending) return
  selectedFramePending = true
  requestAnimationFrame(() => {
    selectedFramePending = false
    selectedTick.value++
  })
}

export function useVideoStream() {
  const cameras = useState<CameraMap>('cameras', () => ({}))
  const cameraOptions = useState<CameraOption[]>('cameraOptions', () => [])
  const selectedCamera = useState<string>('selectedCamera', () => '')
  const isAutoMode = useState<boolean>('isAutoMode', () => false)

  const cameraStatus = useState<CameraStatusMap>('cameraStatus', () => ({}))

  const trackedVehicle = useState<string | null>('trackedVehicle', () => null)
  const { matchThreshold, statusPollInterval } = useSettings()
  const appConfig = useAppConfig()
  
  const backends = resolveBackends(appConfig.backend)
  const wsReconnectTimeout = (appConfig.backend as any)?.wsReconnectTimeout ?? 3000
  const backendCount = backends.length
  const isConnected = computed(() => connectedCount.value === backendCount)

  const videoFeedUrl = (cameraId: string) => {
    const apiUrl = cameraBackend.value[cameraId] ?? backends[0]!.apiUrl
    return `${apiUrl}/video_feed/${cameraId}`
  }

  // Get detections for currently selected camera
  const currentDetections = computed(() => {
    selectedTick.value // re-read the store when the selected camera has new detections
    if (!selectedCamera.value) return []
    return detectionStore[selectedCamera.value] || []
  })

  const fetchCameras = async () => {
    try {
      // a node that is down must not hide the cameras of the others
      const results = await Promise.allSettled(
        backends.map(b => cameraService.fetchCameras(b.apiUrl))
      )

      const merged: CameraMap = {}
      const owners: Record<string, string> = {}
      results.forEach((result, i) => {
        if (result.status === 'fulfilled') {
          camerasByNode[i] = result.value
        } else {
          console.error(`Failed to fetch cameras from ${backends[i]!.apiUrl}:`, result.reason)
        }
        for (const [id, name] of Object.entries(camerasByNode[i] ?? {})) {
          merged[id] = name as string
          owners[id] = backends[i]!.apiUrl
        }
      })

      cameraBackend.value = owners
      cameras.value = merged
      updateCameraOptions()
      const firstOnlineCamera = cameraOptions.value.find(cam => !cam.disabled)
      if (firstOnlineCamera && !selectedCamera.value) {
        selectedCamera.value = firstOnlineCamera.value
      }
    } catch (err) {
      console.error('Failed to fetch cameras:', err)
    }
  }

  const fetchCameraStatus = async () => {
    try {
      const results = await Promise.allSettled(
        backends.map(b => cameraService.fetchCameraStatus(b.apiUrl))
      )

      const status: CameraStatusMap = {}
      results.forEach((result, i) => {
        if (result.status === 'fulfilled') {
          statusByNode[i] = result.value
        } else {
          // node unreachable: show its cameras as offline
          statusByNode[i] = Object.fromEntries(
            Object.keys(camerasByNode[i] ?? {}).map(id => [id, false])
          )
        }
        Object.assign(status, statusByNode[i])
      })
      // polled every couple of seconds; only touch reactive state when something changed
      const statusJson = JSON.stringify(status)
      if (statusJson === lastStatusJson) return
      lastStatusJson = statusJson
      cameraStatus.value = status
      updateCameraOptions()
    } catch (err) {
      console.error('Failed to fetch camera status:', err)
    }
  }

  const updateCameraOptions = () => {
    cameraOptions.value = Object.entries(cameras.value).map(([id, name]) => ({
      value: id,
      label: name as string,
      disabled: cameraStatus.value[id] === false  // Disable if explicitly offline
    }))
  }

  const updateConnectedCount = () => {
    connectedCount.value = sockets.filter(s => s?.readyState === WebSocket.OPEN).length
  }

  const connectWs = () => {
    if (import.meta.server) return
    wsShouldReconnect = true
    backends.forEach((_, i) => connectNode(i))
  }

  const connectNode = (i: number) => {
    // never keep two sockets per node open (e.g. after a hot reload): each one doubles the message load
    const previous = sockets[i]
    if (previous) {
      previous.onclose = null
      previous.close()
    }
    const socket = new WebSocket(backends[i]!.wsUrl)
    sockets[i] = socket

    socket.onopen = () => {
      updateConnectedCount()
      // the list is only fetched once on mount; if this node was not up yet its cameras are missing
      if (!camerasByNode[i]) fetchCameras()
    }

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        const camUuid = data.camera_uuid
        const detections = data.detections || []

        detectionStore[camUuid] = detections
        scheduleMapFlush()
        if (camUuid === selectedCamera.value) scheduleSelectedFlush()

        // Handle Auto Mode Logic
        if (isAutoMode.value) {
          let bestScore = -1
          
          for (const det of detections) {
            if (det.matches && det.matches.length > 0) {
              for (const match of det.matches) {
                const uuid = match[0]
                const score = match[1]
                
                // If tracking a specific vehicle, ONLY look at scores for that vehicle
                if (trackedVehicle.value && uuid !== trackedVehicle.value) {
                  continue
                }
                
                if (score > bestScore) {
                  bestScore = score
                }
              }
            }
          }
          
          if (bestScore >= matchThreshold.value) {
            if (selectedCamera.value !== camUuid) {
              selectedCamera.value = camUuid
            }
          }
        }

      } catch (e) {
        console.error('Failed to parse WS message:', e)
      }
    }
    
    socket.onclose = () => {
      if (sockets[i] === socket) sockets[i] = null
      updateConnectedCount()
      if (wsShouldReconnect) setTimeout(() => connectNode(i), wsReconnectTimeout)
    }
  }

  const disconnectWs = () => {
    wsShouldReconnect = false
    sockets.forEach(socket => socket?.close())
  }

  const startStatusPolling = () => {
    if (import.meta.server) return
    fetchCameraStatus() // Initial fetch
    statusPollingInterval = setInterval(fetchCameraStatus, statusPollInterval.value)
  }

  const stopStatusPolling = () => {
    if (statusPollingInterval) clearInterval(statusPollingInterval)
  }

  // React to poll interval changes dynamically
  watch(statusPollInterval, (newInterval) => {
    if (import.meta.server) return
    if (statusPollingInterval) {
      stopStatusPolling()
      statusPollingInterval = setInterval(fetchCameraStatus, newInterval)
    }
  })

  return {
    trackedVehicle,
    cameras,
    cameraOptions,
    selectedCamera,
    isAutoMode,
    isConnected,
    connectedCount,
    backendCount,
    videoFeedUrl,
    currentDetections,
    cameraDetections,
    cameraStatus,
    fetchCameras,
    fetchCameraStatus,
    startStatusPolling,
    stopStatusPolling,
    connectWs,
    disconnectWs
  }
}
