<!-- for the *contents* of the main panel -->

<template>
  <div class="flex flex-col h-full w-full">
    <!-- Toolbar -->
    <div class="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-800">
      <div class="flex items-center gap-4">
        <USelectMenu
          v-model="selectedCamera"
          :options="cameraOptions"
          option-attribute="label"
          value-attribute="value"
          placeholder="Select a camera"
          class="w-64"
          :filter="{ icon: 'i-lucide-search', placeholder: 'Search cameras...' }"
        >
          <template #trailing>
            <UToggle v-model="isAutoMode" size="sm" class="ml-2" />
            <span class="text-xs text-gray-500 mr-2">Auto</span>
          </template>
        </USelectMenu>
      </div>
      <div>
        <UBadge v-if="isConnected" color="green">WS Connected</UBadge>
        <UBadge v-else color="red">WS Disconnected</UBadge>
      </div>
    </div>

    <!-- Video Feed & Overlays display -->
    <div class="relative flex-1 bg-black overflow-hidden flex items-center justify-center">
      <div v-if="selectedCamera" class="relative inline-block" ref="videoContainerRef">
        <!-- MJPEG Video Stream -->
        <img
          ref="videoElement"
          :src="`http://localhost:8765/video_feed/${selectedCamera}`"
          alt="Live Camera Feed"
          class="max-h-full max-w-full object-contain"
          @load="onVideoLoad"
        />

        <!-- Bounding Box Overlays -->
        <div
          v-for="(det, idx) in currentDetections"
          :key="idx"
          class="absolute border-2 transition-colors"
          :class="[isTracked(det) ? 'border-primary-500 z-20' : 'border-red-500 z-10']"
          :style="getBboxStyle(det.box)"
        >
          <!-- Tooltip / Label for matches -->
          <div 
            class="absolute -top-6 left-0 text-white text-xs px-1 whitespace-nowrap rounded"
            :class="[isTracked(det) ? 'bg-primary-500' : 'bg-red-500']"
          >
            ID: {{ det.vehicle_id }}
            <span v-if="getTopMatchScore(det) !== null">
              | Score: {{ getTopMatchScore(det).toFixed(2) }}
            </span>
          </div>
        </div>
      </div>
      
      <div v-else class="text-gray-500">
        Please select a camera stream.
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onBeforeUnmount } from 'vue'

const {
  trackedVehicle,
  cameraOptions,
  selectedCamera,
  isAutoMode,
  isConnected,
  currentDetections,
  fetchCameras,
  connectWs,
  disconnectWs
} = useVideoStream()

const videoElement = ref<HTMLImageElement | null>(null)
const videoContainerRef = ref<HTMLDivElement | null>(null)
const naturalVideoSize = ref({ width: 1920, height: 1080 })

onMounted(async () => {
  await fetchCameras()
  connectWs()
})

onBeforeUnmount(() => {
  disconnectWs()
})

function onVideoLoad(e: Event) {
  const target = e.target as HTMLImageElement
  naturalVideoSize.value.width = target.naturalWidth
  naturalVideoSize.value.height = target.naturalHeight
}

function getBboxStyle(box: number[]) {
  if (!box || box.length !== 4) return {}

  const [x1, y1, x2, y2] = box
  const w = naturalVideoSize.value.width
  const h = naturalVideoSize.value.height

  return {
    left: `${(x1 / w) * 100}%`,
    top: `${(y1 / h) * 100}%`,
    width: `${((x2 - x1) / w) * 100}%`,
    height: `${((y2 - y1) / h) * 100}%`
  }
}

function isTracked(det: any) {
  if (!trackedVehicle.value || !det.matches) return false
  return det.matches.some((m: any) => m[0] === trackedVehicle.value)
}

function getTopMatchScore(det: any) {
  if (!det.matches || det.matches.length === 0) return null
  if (trackedVehicle.value) {
    const m = det.matches.find((m: any) => m[0] === trackedVehicle.value)
    if (m) return m[1]
  }
  return det.matches[0][1]
}
</script>