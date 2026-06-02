<!-- for the *contents* of the main panel -->

<template>
  <div class="flex flex-col h-full w-full">
    <!-- Video Feed & Overlays display -->
    <div class="relative flex-1 bg-black overflow-hidden flex items-center justify-center" ref="videoWrapperRef">
      <div v-if="selectedCamera" class="relative inline-block" :style="containerStyle" ref="videoContainerRef">
        <!-- MJPEG Video Stream -->
        <img
          ref="videoElement"
          :src="`${backend.apiUrl}/video_feed/${selectedCamera}`"
          alt="Live Camera Feed"
          style="width: 100%; height: 100%; display: block;"
          @load="onVideoLoad"
        />

        <!-- Bounding Box Overlays -->
        <div
          v-for="(det, idx) in currentDetections"
          :key="idx"
          class="absolute border-2 transition-colors"
          :class="[isHighestMatch(det) ? 'border-red-500 z-20' : 'border-green-500 z-10']"
          :style="getBboxStyle(det.box)"
        >
          <!-- Tooltip / Label for matches -->
          <div 
            class="absolute -top-6 left-0 text-white text-xs px-1 whitespace-nowrap rounded"
            :class="[isHighestMatch(det) ? 'bg-red-500' : 'bg-green-500']"
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
const appConfig = useAppConfig()
const backend = (appConfig.backend as any) || { apiUrl: 'http://localhost:8765' }

const {
  trackedVehicle,
  selectedCamera,
  currentDetections,
  fetchCameras,
  connectWs,
  disconnectWs,
  startStatusPolling,
  stopStatusPolling
} = useVideoStream()

const {
  videoElement,
  videoContainerRef,
  videoWrapperRef,
  containerStyle,
  updateNaturalVideoSize: onVideoLoad,
  getBboxStyle,
  getTopMatchScore,
  isHighestMatch
} = useMainpanelVideo(trackedVehicle, currentDetections)

onMounted(async () => {
  await fetchCameras()
  connectWs()
  startStatusPolling()
})

onBeforeUnmount(() => {
  disconnectWs()
  stopStatusPolling()
})
</script>