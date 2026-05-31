<template>
  <div class="flex flex-col h-full w-full">
    <UDashboardNavbar title="Live Map" />
    <div class="flex-1 relative bg-gray-900">
      <div id="osm-map" class="absolute inset-0 z-0"></div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, watch } from 'vue'

const { cameraDetections, trackedVehicle } = useVideoStream()
const THRESHOLD = 0.4

const cameraLocations = ref<Record<string, {name: string, lat: number, lon: number}>>({})
const markers: Record<string, any> = {}
let mapInstance: any = null
let L: any = null

onMounted(async () => {
  // import leaflet to prevent SSR "window is not defined" errors
  L = (await import('leaflet')).default
  await import('leaflet/dist/leaflet.css')

  try {
    const res = await fetch('/pbc_coords.csv')
    if (res.ok) {
      const csvText = await res.text()
      const lines = csvText.trim().split('\n')
      const locations: Record<string, {name: string, lat: number, lon: number}> = {}
      
      for (let i = 1; i < lines.length; i++) {
        const line = lines[i].trim()
        if (!line) continue
        
        const parts = line.split(',')
        if (parts.length >= 4) {
          const uuid = parts[0].trim()
          const lon = parseFloat(parts.pop()!.trim())
          const lat = parseFloat(parts.pop()!.trim())
          const name = parts.slice(1).join(',').trim()
          
          locations[uuid] = { name, lat, lon }
        }
      }
      cameraLocations.value = locations
    }
  } catch (err) {
    console.error('Failed to load camera locations:', err)
  }

  const firstLoc = Object.values(cameraLocations.value)[0]
  const startLat = firstLoc?.lat ?? 26.627
  const startLon = firstLoc?.lon ?? -80.084
  const startZoom = 10

  mapInstance = L.map('osm-map').setView([startLat, startLon], startZoom)

  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
    maxZoom: 19
  }).addTo(mapInstance)

  for (const [uuid, loc] of Object.entries(cameraLocations.value)) {
    markers[uuid] = L.marker([loc.lat, loc.lon]).addTo(mapInstance)
  }

  updateMarkers()
})

watch([cameraDetections, trackedVehicle], () => {
  updateMarkers()
}, { deep: true })

function updateMarkers() {
  if (!mapInstance || !L) return

  let globalBestScore = -1
  let globalBestUuid: string | null = null
  const currentScores: Record<string, number> = {}

  // 1. Calculate the highest score for each camera
  for (const uuid of Object.keys(cameraLocations.value)) {
    const detections = cameraDetections.value[uuid] || []
    let cameraBest = 0

    for (const det of detections) {
      if (!det.matches || det.matches.length === 0) continue

      for (const match of det.matches) {
        const [matchUuid, matchScore] = match
        
        if (trackedVehicle.value && matchUuid !== trackedVehicle.value) continue
        
        if (matchScore > cameraBest) {
          cameraBest = matchScore
        }
      }
    }

    currentScores[uuid] = cameraBest

    if (cameraBest > globalBestScore) {
      globalBestScore = cameraBest
      globalBestUuid = uuid
    }
  }

  // 2. Update Map Markers
  for (const [uuid, marker] of Object.entries(markers)) {
    const score = currentScores[uuid] || 0
    const isGlobalBest = (uuid === globalBestUuid) && (score >= THRESHOLD)

    // Calculate darkness of green based on score (0 to 1)
    // 40 is a dark green baseline, 255 is bright green
    const greenIntensity = Math.max(40, Math.floor(score * 255))
    
    let backgroundColor = `rgb(0, ${greenIntensity}, 0)`
    let borderColor = `rgb(0, ${Math.min(255, greenIntensity + 50)}, 0)`

    if (isGlobalBest) {
      backgroundColor = 'rgb(239, 68, 68)' // tailwind red-500
      borderColor = 'rgb(248, 113, 113)'   // tailwind red-400
    }

    // use Leaflet DivIcon for full CSS control over the marker

    let icon_size = 24

    const iconHtml = `
      <div style="
        background-color: ${backgroundColor};
        color: white;
        border-radius: 50%;
        width: ${icon_size}px;
        height: ${icon_size}px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 11px;
        font-weight: 600;
        border: 2px solid ${borderColor};
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
        transition: all 0.3s ease;
      ">
        ${score > 0 ? score.toFixed(2) : '-'}
      </div>
    `

    marker.setIcon(L.divIcon({
      html: iconHtml,
      className: 'bg-transparent border-none',  // override default Leaflet classes
      iconSize: [36, 36],
      iconAnchor: [18, 18],                     // center the icon on coords
      tooltipAnchor: [0, -18]
    }))

    if (cameraLocations.value[uuid]) {
      marker.bindTooltip(cameraLocations.value[uuid].name, { 
        direction: 'top', 
        offset: [0, -10],
        opacity: 0.9
      })
    }
  }
}
</script>

<style>
.leaflet-container {
  z-index: 10 !important;
}
</style>