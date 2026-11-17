<!-- database view of the main panel: one vehicle's reference images, tracking settings and recorded sightings -->

<template>
  <div class="flex flex-col h-full w-full text-sm">
    <div v-if="!vehicle" class="flex-1 flex items-center justify-center text-dimmed">
      {{ vehicles.length === 0 ? 'No vehicles yet' : 'No vehicle selected' }}
    </div>

    <template v-else>
      <!-- Info bar -->
      <div class="flex items-center gap-3 h-12 px-4 border-b border-default shrink-0">
        <UInput
          v-model="name"
          variant="none"
          placeholder="Unnamed vehicle"
          aria-label="Vehicle name"
          class="flex-1 min-w-0"
          :ui="{ base: 'px-0 font-medium truncate' }"
          @blur="saveName"
          @keydown.enter="($event.target as HTMLInputElement).blur()"
        />
        <span class="text-xs text-dimmed font-mono">{{ vehicle.uuid.split('-')[0] }}</span>
        <UButton
          icon="i-lucide-trash-2"
          color="neutral"
          variant="ghost"
          size="xs"
          :loading="deleting === vehicle.uuid"
          aria-label="Delete vehicle"
          @click="removeVehicle"
        />
      </div>

      <div class="flex-1 overflow-y-auto">
        <!-- Reference images -->
        <section class="px-4 py-3 border-b border-default space-y-2">
          <h3 class="section-label">Reference images</h3>
          <div class="flex flex-wrap gap-2">
            <div
              v-for="image in vehicle.images"
              :key="image"
              class="group relative size-24 rounded-md overflow-hidden border border-default bg-black"
            >
              <img :src="vehicleService.imageUrl(apiUrl, vehicle.uuid, image)" alt="" class="size-full object-cover" />
              <UButton
                icon="i-lucide-x"
                color="neutral"
                variant="solid"
                size="xs"
                class="absolute top-1 right-1 opacity-0 group-hover:opacity-100 focus-visible:opacity-100"
                aria-label="Remove image"
                @click="deleteVehicleImage(vehicle.uuid, image)"
              />
            </div>

            <button
              type="button"
              class="size-24 rounded-md border border-dashed border-accented text-dimmed hover:text-default hover:border-inverted/40 flex items-center justify-center cursor-pointer disabled:cursor-wait"
              :disabled="isAddingImages"
              aria-label="Add images"
              @click="fileInput?.click()"
            >
              <UIcon :name="isAddingImages ? 'i-lucide-loader-circle' : 'i-lucide-plus'" class="size-5" :class="{ 'animate-spin': isAddingImages }" />
            </button>
            <input ref="fileInput" type="file" multiple accept="image/jpeg,image/png,image/webp" class="hidden" @change="addImages" />
          </div>
          <p v-if="vehicle.images.length === 0" class="text-xs text-muted">
            Without a reference image this vehicle cannot be matched.
          </p>
        </section>

        <!-- Tracking -->
        <section class="px-4 py-3 border-b border-default space-y-2">
          <h3 class="section-label">Tracking</h3>
          <label class="flex items-center justify-between gap-3 cursor-pointer">
            <span class="text-muted">Record sightings while the server runs</span>
            <USwitch :model-value="vehicle.tracking" size="sm" @update:model-value="updateVehicle(vehicle.uuid, { tracking: $event })" />
          </label>
          <div class="flex items-center justify-between gap-3">
            <span class="text-muted">Minimum match score to record</span>
            <UInput v-model.number="threshold" type="number" step="0.05" min="0" max="1" size="xs" class="w-20" @change="saveThreshold" />
          </div>
        </section>

        <!-- Sightings -->
        <section class="px-4 py-3">
          <div class="flex items-center justify-between h-6 mb-2">
            <h3 class="section-label">
              Sightings<template v-if="vehicle.sightings > 0"> · {{ isFiltered ? `${total} of ${vehicle.sightings}` : total }}</template>
            </h3>
            <div v-if="vehicle.sightings > 0" class="flex items-center gap-3">
              <label class="flex items-center gap-2 text-xs text-muted">
                Show score ≥
                <UInput v-model.number="minScoreInput" type="number" step="0.05" min="0" max="1" size="xs" class="w-20" @change="applyMinScore" />
              </label>
              <UButton label="Clear all" color="neutral" variant="ghost" size="xs" @click="clearSightings" />
            </div>
          </div>

          <p v-if="sightings.length === 0" class="py-6 text-center text-muted">
            <template v-if="vehicle.sightings > 0">No sightings scored {{ sightingMinScore.toFixed(2) }} or higher.</template>
            <template v-else>{{ vehicle.tracking ? 'No sightings recorded yet.' : 'Tracking is off for this vehicle.' }}</template>
          </p>

          <table v-else class="w-full">
            <thead>
              <tr class="text-left text-xs text-dimmed">
                <th class="font-normal pb-1.5 w-24">Capture</th>
                <th class="font-normal pb-1.5">Location</th>
                <th class="font-normal pb-1.5">Time</th>
                <th class="font-normal pb-1.5 text-right">Score</th>
                <th class="w-8" />
              </tr>
            </thead>
            <tbody>
              <tr v-for="sighting in sightings" :key="sighting.id" class="group border-t border-default">
                <td class="py-1.5 pr-3">
                  <a :href="vehicleService.sightingImageUrl(apiUrl, sighting)" target="_blank" class="block h-14 w-20 rounded bg-black overflow-hidden">
                    <img :src="vehicleService.sightingImageUrl(apiUrl, sighting)" alt="Captured vehicle" loading="lazy" class="size-full object-contain" />
                  </a>
                </td>
                <td class="py-1.5 pr-3">
                  <button type="button" class="text-left hover:underline cursor-pointer" title="Show this camera live" @click="showCamera(sighting.camera_uuid)">
                    {{ sighting.camera_name || sighting.camera_uuid }}
                  </button>
                </td>
                <td class="py-1.5 pr-3 text-muted whitespace-nowrap">{{ formatTime(sighting.seen_at) }}</td>
                <td class="py-1.5 text-right font-mono text-xs">{{ sighting.score.toFixed(2) }}</td>
                <td class="py-1.5 text-right">
                  <UButton
                    icon="i-lucide-trash-2"
                    color="neutral"
                    variant="ghost"
                    size="xs"
                    class="opacity-0 group-hover:opacity-100 focus-visible:opacity-100"
                    aria-label="Delete sighting"
                    @click="deleteSighting(sighting.id)"
                  />
                </td>
              </tr>
            </tbody>
          </table>

          <div v-if="sightings.length < total" class="pt-3 text-center">
            <UButton label="Show more" color="neutral" variant="ghost" size="xs" @click="limit += PAGE_SIZE" />
          </div>
        </section>
      </div>
    </template>
  </div>
</template>

<script setup lang="ts">
import { vehicleService } from '../services/vehicleService'
import type { Sighting } from '../types/vehicle'

const PAGE_SIZE = 50
const REFRESH_MS = 5000

const {
  apiUrl,
  databaseVehicle,
  vehicles,
  deleting,
  fetchVehicles,
  deleteVehicleRecord,
  updateVehicle,
  addVehicleImages,
  deleteVehicleImage
} = useVehicles()
const { view, sightingMinScore } = useSettings()
const { selectedCamera, isAutoMode } = useVideoStream()

const vehicle = computed(() => vehicles.value.find(v => v.uuid === databaseVehicle.value) ?? null)

// ---- name and threshold are edited in place and saved when the field is left

const name = ref('')
const threshold = ref(0.5)

// the vehicle list is refetched every few seconds; only reset a field when its saved value changes,
// so a refresh does not overwrite what is being typed
watch(() => `${vehicle.value?.uuid}|${vehicle.value?.name}`, () => {
  name.value = vehicle.value?.name ?? ''
}, { immediate: true })
watch(() => `${vehicle.value?.uuid}|${vehicle.value?.threshold}`, () => {
  threshold.value = vehicle.value?.threshold ?? 0.5
}, { immediate: true })

const saveName = () => {
  const v = vehicle.value
  if (!v || name.value.trim() === (v.name ?? '')) return
  updateVehicle(v.uuid, { name: name.value.trim() })
}

const saveThreshold = () => {
  const v = vehicle.value
  if (!v) return
  const value = Math.min(1, Math.max(0, Number(threshold.value)))
  if (Number.isNaN(value)) {
    threshold.value = v.threshold
    return
  }
  threshold.value = value
  if (value !== v.threshold) updateVehicle(v.uuid, { threshold: value })
}

const removeVehicle = async () => {
  if (!vehicle.value) return
  if (!confirm('Delete this vehicle and its recorded sightings?')) return
  await deleteVehicleRecord(vehicle.value.uuid)
}

// ---- reference images

const fileInput = ref<HTMLInputElement | null>(null)
const isAddingImages = ref(false)

const addImages = async () => {
  if (!vehicle.value || !fileInput.value) return
  isAddingImages.value = true
  await addVehicleImages(vehicle.value.uuid, fileInput.value.files)
  fileInput.value.value = ''
  isAddingImages.value = false
}

// ---- sightings

const sightings = ref<Sighting[]>([])
const total = ref(0)
const limit = ref(PAGE_SIZE)

// viewing filter only: everything at or above the recording score stays in the database
const minScoreInput = ref(sightingMinScore.value)
const isFiltered = computed(() => sightingMinScore.value > 0)

const applyMinScore = () => {
  const value = Math.min(1, Math.max(0, Number(minScoreInput.value)))
  minScoreInput.value = Number.isNaN(value) ? sightingMinScore.value : value
  sightingMinScore.value = minScoreInput.value
}

const fetchSightings = async () => {
  const uuid = databaseVehicle.value
  if (!uuid) {
    sightings.value = []
    total.value = 0
    return
  }
  try {
    const page = await vehicleService.fetchSightings(apiUrl, uuid, limit.value, sightingMinScore.value)
    if (uuid !== databaseVehicle.value) return // another vehicle was opened meanwhile
    sightings.value = page.items
    total.value = page.total
  } catch (err) {
    console.error('Failed to fetch sightings:', err)
  }
}

const deleteSighting = async (id: number) => {
  try {
    await vehicleService.deleteSighting(apiUrl, id)
  } catch (err) {
    console.error('Failed to delete sighting:', err)
  }
  await refresh()
}

const clearSightings = async () => {
  if (!vehicle.value) return
  const hidden = isFiltered.value ? ', including the ones hidden by the score filter' : ''
  if (!confirm(`Delete all ${vehicle.value.sightings} sightings of this vehicle${hidden}?`)) return
  try {
    await vehicleService.clearSightings(apiUrl, vehicle.value.uuid)
  } catch (err) {
    console.error('Failed to clear sightings:', err)
  }
  await refresh()
}

const showCamera = (cameraUuid: string) => {
  isAutoMode.value = false
  selectedCamera.value = cameraUuid
  view.value = 'live'
}

const formatTime = (seenAt: number) =>
  new Date(seenAt * 1000).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'medium' })

// sightings are recorded on the server whether or not this page is open; keep the view current
const refresh = () => Promise.all([fetchVehicles(), fetchSightings()])
let refreshTimer: ReturnType<typeof setInterval> | null = null

watch(databaseVehicle, () => {
  sightings.value = []
  total.value = 0
  limit.value = PAGE_SIZE
  fetchSightings()
})
watch(sightingMinScore, () => {
  limit.value = PAGE_SIZE
  fetchSightings()
})
watch(limit, fetchSightings)

onMounted(async () => {
  await refresh()
  if (!vehicle.value && vehicles.value.length > 0) {
    databaseVehicle.value = vehicles.value[0]!.uuid
  }
  refreshTimer = setInterval(refresh, REFRESH_MS)
})

onBeforeUnmount(() => {
  if (refreshTimer) clearInterval(refreshTimer)
})
</script>
