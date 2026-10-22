<template>
  <div class="flex flex-col h-full text-sm">
    <!-- Header -->
    <div class="flex items-center justify-between h-12 px-4 border-b border-default shrink-0">
      <span class="font-semibold tracking-tight">cartracker</span>
      <span class="flex items-center gap-1.5 text-xs text-muted">
        <span class="size-1.5 rounded-full" :class="isConnected ? 'bg-green-500' : 'bg-red-500'" />
        {{ isConnected ? 'Connected' : 'Disconnected' }}
      </span>
    </div>

    <!-- Camera -->
    <section class="px-4 py-3 border-b border-default space-y-2">
      <h3 class="section-label">Camera</h3>
      <USelectMenu
        v-model="selectedCameraItem"
        :items="cameraOptions"
        :disabled="isAutoMode"
        class="w-full"
        size="sm"
        placeholder="Select a camera"
      />
      <label class="flex items-center justify-between cursor-pointer">
        <span class="text-muted">Follow best match</span>
        <USwitch v-model="isAutoMode" size="sm" />
      </label>
    </section>

    <!-- Vehicles -->
    <section class="flex-1 min-h-0 flex flex-col">
      <div class="flex items-center justify-between px-4 pt-3 pb-2">
        <h3 class="section-label">Vehicles</h3>
        <UButton icon="i-lucide-plus" size="xs" color="neutral" variant="ghost" aria-label="Add vehicle" @click="isModalOpen = true" />
      </div>

      <div class="flex-1 overflow-y-auto px-2 pb-2">
        <p v-if="vehicleItems.length === 0" class="px-2 py-6 text-center text-muted">
          No vehicles yet.
        </p>

        <ul v-else class="space-y-px">
          <li
            v-for="item in vehicleItems"
            :key="item.value"
            class="group flex items-center gap-2 px-2 py-1.5 rounded-md cursor-pointer"
            :class="trackedVehicle === item.value ? 'bg-elevated' : 'hover:bg-elevated/50'"
            @click="toggleTrack(item.value)"
          >
            <span
              class="size-2 rounded-full shrink-0"
              :class="trackedVehicle === item.value ? 'bg-primary' : 'bg-accented'"
            />
            <div class="min-w-0 flex-1">
              <div class="truncate">{{ item.label }}</div>
              <div class="truncate text-xs text-dimmed font-mono">{{ item.value.split('-')[0] }}</div>
            </div>
            <UButton
              icon="i-lucide-trash-2"
              color="neutral"
              variant="ghost"
              size="xs"
              class="opacity-0 group-hover:opacity-100"
              :loading="deleting === item.value"
              aria-label="Delete vehicle"
              @click.stop="deleteVehicle(item.value)"
            />
          </li>
        </ul>
      </div>
    </section>

    <!-- Settings -->
    <section class="px-4 py-3 border-t border-default space-y-2">
      <h3 class="section-label">Settings</h3>
      <div class="flex items-center justify-between gap-3">
        <span class="text-muted">Match threshold</span>
        <UInput v-model.number="matchThreshold" type="number" step="0.05" min="0" max="1" size="xs" class="w-20" />
      </div>
      <div class="flex items-center justify-between gap-3">
        <span class="text-muted">Poll interval (ms)</span>
        <UInput v-model.number="statusPollInterval" type="number" step="100" min="500" size="xs" class="w-20" />
      </div>
    </section>

    <UModal v-model:open="isModalOpen" title="Add vehicle">
      <template #body>
        <form class="space-y-4" @submit.prevent="uploadVehicle">
          <UFormField label="Name" hint="Optional">
            <UInput v-model="vehicleName" class="w-full" />
          </UFormField>
          <UFormField label="Photos">
            <input
              ref="fileInput"
              type="file"
              multiple
              accept="image/*"
              class="block w-full text-sm text-muted file:mr-3 file:px-3 file:py-1.5 file:rounded-md file:border file:border-default file:bg-elevated file:text-default file:text-sm file:cursor-pointer"
            />
          </UFormField>
          <div class="flex justify-end gap-2">
            <UButton color="neutral" variant="ghost" label="Cancel" @click="isModalOpen = false" />
            <UButton type="submit" label="Upload" :loading="isUploading" />
          </div>
        </form>
      </template>
    </UModal>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'

const { matchThreshold, statusPollInterval } = useSettings()

const {
  cameraOptions,
  selectedCamera,
  isAutoMode,
  isConnected
} = useVideoStream()

const selectedCameraItem = computed({
  get: () => cameraOptions.value.find(cam => cam.value === selectedCamera.value) ?? null,
  set: (cam: { value: string; label: string } | null) => {
    selectedCamera.value = cam?.value ?? ''
  }
})

const {
  trackedVehicle,
  vehicleItems,
  isUploading,
  deleting,
  fetchVehicles,
  uploadVehicleFiles,
  deleteVehicleRecord
} = useVehicles()

const isModalOpen = ref(false)
const fileInput = ref<HTMLInputElement | null>(null)
const vehicleName = ref('')

onMounted(() => {
  fetchVehicles()
})

const uploadVehicle = async () => {
  const success = await uploadVehicleFiles(fileInput.value?.files, vehicleName.value)
  if (success) {
    isModalOpen.value = false
    vehicleName.value = ''
    if (fileInput.value) fileInput.value.value = ''
  }
}

const deleteVehicle = async (uuid: string) => {
  if (!confirm('Are you sure you want to delete this vehicle?')) return
  await deleteVehicleRecord(uuid)
}

const toggleTrack = (uuid: string) => {
  if (trackedVehicle.value === uuid) {
    trackedVehicle.value = null
  } else {
    trackedVehicle.value = uuid
  }
}
</script>
