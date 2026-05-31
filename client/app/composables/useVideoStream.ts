import { ref, computed } from 'vue'

const AUTO_THRESHOLD = 0.4
const STATUS_POLL_INTERVAL = 2000 // Poll every 2 seconds

const cameras = ref<Record<string, string>>({})
const cameraOptions = ref<{ value: string; label: string; disabled?: boolean }[]>([])
const selectedCamera = ref('')
const isAutoMode = ref(false)

const isConnected = ref(false)
let ws: WebSocket | null = null
let statusPollingInterval: NodeJS.Timeout | null = null

const cameraDetections = ref<Record<string, any[]>>({})
const cameraStatus = ref<Record<string, boolean>>({})  // camera_uuid -> is_online

export function useVideoStream() {
  const trackedVehicle = useState<string | null>('trackedVehicle', () => null)

  // Get detections for currently selected camera
  const currentDetections = computed(() => {
    if (!selectedCamera.value) return []
    return cameraDetections.value[selectedCamera.value] || []
  })

  const fetchCameras = async () => {
    try {
      const res = await fetch('http://localhost:8765/cameras')
      if (res.ok) {
        cameras.value = await res.json()
        updateCameraOptions()
        const firstOnlineCamera = cameraOptions.value.find(cam => !cam.disabled)
        if (firstOnlineCamera) {
          selectedCamera.value = firstOnlineCamera.value
        }
      }
    } catch (err) {
      console.error('Failed to fetch cameras:', err)
    }
  }

  const fetchCameraStatus = async () => {
    try {
      const res = await fetch('http://localhost:8765/camera_status')
      if (res.ok) {
        cameraStatus.value = await res.json()
        updateCameraOptions()

        /*
        if (selectedCamera.value && !cameraStatus.value[selectedCamera.value]) {
          const firstOnlineCamera = cameraOptions.value.find(cam => !cam.disabled)
          if (firstOnlineCamera) {
            selectedCamera.value = firstOnlineCamera.value
          }
        }
        */
      }
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
    ws = new WebSocket('ws://localhost:8765/ws')
    
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
          
          if (bestScore >= AUTO_THRESHOLD) {
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
      setTimeout(connectWs, 3000)
    }
  }

  const disconnectWs = () => {
    if (ws) ws.close()
  }

  const startStatusPolling = () => {
    fetchCameraStatus() // Initial fetch
    statusPollingInterval = setInterval(fetchCameraStatus, STATUS_POLL_INTERVAL)
  }

  const stopStatusPolling = () => {
    if (statusPollingInterval) clearInterval(statusPollingInterval)
  }

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
