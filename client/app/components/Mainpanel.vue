<!-- for the *contents* of the main panel -->

<template>
  <div class="flex flex-col h-full w-full">
    <!-- Toolbar -->
    <div class="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-800">
      <div class="flex items-center gap-4">
        <USelectMenu
          v-model="selectedCameraItem"
          :items="cameraOptions"
          class="w-64"
          placeholder="Select a camera"
        />
        <USwitch v-model="isAutoMode" size="sm" class="ml-4 -mr-2" />
            <span class="text-xs text-gray-500 mr-2 ml-0">Auto</span>
      </div>
      <div>
        <UBadge v-if="isConnected" color="green">WS Connected</UBadge>
        <UBadge v-else color="red">WS Disconnected</UBadge>
      </div>
    </div>

    <!-- Video Feed & Overlays display -->
    <div class="relative flex-1 bg-black overflow-hidden flex items-center justify-center" ref="videoWrapperRef">
      <div v-if="selectedCamera" class="relative inline-block" :style="containerStyle" ref="videoContainerRef">
        <!-- MJPEG Video Stream -->
        <img
          ref="videoElement"
          :src="`http://localhost:8765/video_feed/${selectedCamera}`"
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
import { computed } from 'vue'

const {
  trackedVehicle,
  cameraOptions,
  selectedCamera,
  isAutoMode,
  isConnected,
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

const selectedCameraItem = computed({
  get: () => cameraOptions.value.find(cam => cam.value === selectedCamera.value) ?? null,
  set: (cam: { value: string; label: string } | null) => {
    selectedCamera.value = cam?.value ?? ''
  }
})

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