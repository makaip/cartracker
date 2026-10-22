<!-- for the *contents* of the main panel -->

<template>
  <div class="flex flex-col h-full w-full">
    <!-- Info bar -->
    <div class="flex items-center gap-4 h-12 px-4 border-b border-default text-sm shrink-0">
      <span class="font-medium truncate">{{ cameraName || 'No camera' }}</span>
      <span v-if="isAutoMode" class="text-xs text-primary">auto</span>
      <span class="ml-auto text-muted text-xs whitespace-nowrap">
        {{ currentDetections.length }} {{ currentDetections.length === 1 ? 'detection' : 'detections' }}
      </span>
      <span class="text-muted text-xs whitespace-nowrap">
        Tracking: <span class="text-default">{{ trackedLabel ?? 'any' }}</span>
      </span>
    </div>

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
          class="absolute border"
          :class="[isHighestMatch(det) ? 'border-red-500 border-2 z-20' : 'border-green-400/80 z-10']"
          :style="getBboxStyle(det.box)"
        >
          <!-- Label for matches -->
          <div
            class="absolute bottom-full -left-px mb-px px-1 text-[11px] leading-4 font-mono whitespace-nowrap"
            :class="[isHighestMatch(det) ? 'bg-red-500 text-white' : 'bg-green-400/80 text-black']"
          >
            {{ det.vehicle_id }}<template v-if="getTopMatchScore(det) !== null"> · {{ getTopMatchScore(det).toFixed(2) }}</template>
          </div>
        </div>
      </div>

      <div v-else class="text-sm text-dimmed">
        No camera selected
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
const appConfig = useAppConfig()
const backend = (appConfig.backend as any) || { apiUrl: 'http://localhost:8765' }

const {
  trackedVehicle,
  cameras,
  isAutoMode,
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
} = useMainPanelVideo(trackedVehicle, currentDetections)

const { vehicleItems } = useVehicles()

const cameraName = computed(() => cameras.value[selectedCamera.value])
const trackedLabel = computed(() =>
  vehicleItems.value.find(v => v.value === trackedVehicle.value)?.label ?? null
)

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