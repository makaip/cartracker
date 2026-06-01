import { ref, computed, onMounted, onBeforeUnmount, type Ref } from 'vue'

type DetectionMatch = [string, number]

interface Detection {
  vehicle_id: string
  box: number[]
  matches?: DetectionMatch[]
}

const THRESHOLD = 0.40

export function useMainpanelVideo(
  trackedVehicle: Ref<string | null>,
  currentDetections: Ref<Detection[]>
) {
  const videoElement = ref<HTMLImageElement | null>(null)
  const videoContainerRef = ref<HTMLDivElement | null>(null)
  const videoWrapperRef = ref<HTMLDivElement | null>(null)
  const naturalVideoSize = ref({ width: 1920, height: 1080 })
  const wrapperSize = ref({ width: 0, height: 0 })

  let resizeObserver: ResizeObserver | null = null

  const containerStyle = computed(() => {
    const { width: nw, height: nh } = naturalVideoSize.value
    const { width: ww, height: wh } = wrapperSize.value

    if (nw === 0 || nh === 0 || ww === 0 || wh === 0) {
      return { width: '100%', height: '100%' }
    }

    const wrapperRatio = ww / wh
    const videoRatio = nw / nh

    if (wrapperRatio > videoRatio) {
      return {
        height: '100%',
        width: `${wh * videoRatio}px`
      }
    }

    return {
      width: '100%',
      height: `${ww / videoRatio}px`
    }
  })

  function updateNaturalVideoSize(e: Event) {
    const target = e.target as HTMLImageElement
    naturalVideoSize.value.width = target.naturalWidth
    naturalVideoSize.value.height = target.naturalHeight
  }

  function getBboxStyle(box: number[]) {
    if (!box || box.length !== 4) return {}

    const [x1, y1, x2, y2] = box
    const { width, height } = naturalVideoSize.value

    return {
      left: `${(x1 / width) * 100}%`,
      top: `${(y1 / height) * 100}%`,
      width: `${((x2 - x1) / width) * 100}%`,
      height: `${((y2 - y1) / height) * 100}%`
    }
  }

  function getTopMatchScore(det: Detection) {
    if (!det.matches || det.matches.length === 0) return null

    if (trackedVehicle.value) {
      const match = det.matches.find((candidate) => candidate[0] === trackedVehicle.value)
      if (match) return match[1]
    }

    return det.matches[0][1]
  }

  function isHighestMatch(det: Detection) {
    if (!trackedVehicle.value || !det.matches) return false

    const score = getTopMatchScore(det)
    if (score === null || score < THRESHOLD) return false

    let highestScore = -1
    let highestDetId: string | null = null

    for (const currentDetection of currentDetections.value) {
      const currentScore = getTopMatchScore(currentDetection)
      if (currentScore !== null && currentScore > highestScore) {
        highestScore = currentScore
        highestDetId = currentDetection.vehicle_id
      }
    }

    return det.vehicle_id === highestDetId
  }

  onMounted(() => {
    if (!videoWrapperRef.value) return

    resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        wrapperSize.value = {
          width: entry.contentRect.width,
          height: entry.contentRect.height
        }
      }
    })

    resizeObserver.observe(videoWrapperRef.value)
  })

  onBeforeUnmount(() => {
    if (resizeObserver && videoWrapperRef.value) {
      resizeObserver.unobserve(videoWrapperRef.value)
    }
  })

  return {
    videoElement,
    videoContainerRef,
    videoWrapperRef,
    containerStyle,
    updateNaturalVideoSize,
    getBboxStyle,
    getTopMatchScore,
    isHighestMatch
  }
}