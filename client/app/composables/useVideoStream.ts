import { computed, watch } from 'vue'
import { cameraService } from '../services/cameraService'
import type { CameraOption, CameraMap, CameraStatusMap, CameraDetectionMap } from '../types/camera'

let ws: WebSocket | null = null
let statusPollingInterval: NodeJS.Timeout | null = null

export function useVideoStream() {
  const cameras = useState<CameraMap>('cameras', () => ({}))
  const cameraOptions = useState<CameraOption[]>('cameraOptions', () => [])
  const selectedCamera = useState<string>('selectedCamera', () => '')
  const isAutoMode = useState<boolean>('isAutoMode', () => false)

  const isConnected = useState<boolean>('isConnected', () => false)
  const cameraDetections = useState<CameraDetectionMap>('cameraDetections', () => ({}))
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
    if (!selectedCamera.value) return []
    return cameraDetections.value[selectedCamera.value] || []
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
      cameraStatus.value = await cameraService.fetchCameraStatus(backend.apiUrl)
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

    ws = new WebSocket(backend.wsUrl)
    
    ws.onopen = () => {
      isConnected.value = true
    }

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        const camUuid = data.camera_uuid
        const detections = data.detections || []

        cameraDetections.value[camUuid] = detections

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
      setTimeout(connectWs, backend.wsReconnectTimeout)
    }
  }

  const disconnectWs = () => {
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
