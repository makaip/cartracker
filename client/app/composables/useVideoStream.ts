import { computed, shallowRef, triggerRef, watch } from 'vue'
import { cameraService } from '../services/cameraService'
import type { CameraOption, CameraMap, CameraStatusMap, CameraDetectionMap } from '../types/camera'

let ws: WebSocket | null = null
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

  const isConnected = useState<boolean>('isConnected', () => false)
  const cameraStatus = useState<CameraStatusMap>('cameraStatus', () => ({}))

  const trackedVehicle = useState<string | null>('trackedVehicle', () => null)
  const { matchThreshold, statusPollInterval } = useSettings()
  const appConfig = useAppConfig()
  
  const backend = (appConfig.backend as any) || {
    apiUrl: 'http://localhost:8765',
    wsUrl: 'ws://localhost:8765/ws',
    wsReconnectTimeout: 3000
  }

  // Get detections for currently selected camera
  const currentDetections = computed(() => {
    selectedTick.value // re-read the store when the selected camera has new detections
    if (!selectedCamera.value) return []
    return detectionStore[selectedCamera.value] || []
  })

  const fetchCameras = async () => {
    try {
      cameras.value = await cameraService.fetchCameras(backend.apiUrl)
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
      const status = await cameraService.fetchCameraStatus(backend.apiUrl)
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

  const connectWs = () => {
    if (import.meta.server) return

    // never keep two sockets open (e.g. after a hot reload): each one doubles the message load
    if (ws) {
      ws.onclose = null
      ws.close()
    }
    wsShouldReconnect = true
    const socket = new WebSocket(backend.wsUrl)
    ws = socket

    ws.onopen = () => {
      isConnected.value = true
      // the list is only fetched once on mount; if the server was not up yet it stays empty
      if (Object.keys(cameras.value).length === 0) fetchCameras()
    }

    ws.onmessage = (event) => {
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
    
    ws.onclose = () => {
      isConnected.value = false
      if (ws === socket) ws = null
      if (wsShouldReconnect) setTimeout(connectWs, backend.wsReconnectTimeout)
    }
  }

  const disconnectWs = () => {
    wsShouldReconnect = false
    if (ws) ws.close()
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
