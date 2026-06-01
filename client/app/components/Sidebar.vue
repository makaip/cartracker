<template>
  <div class="flex flex-col h-full">
    <div class="p-4 border-b border-gray-200 dark:border-gray-800 space-y-4">
      <div class="flex items-center justify-between gap-4">
        <USelectMenu
          v-model="selectedCameraItem"
          :items="cameraOptions"
          class="flex-1"
          placeholder="Select a camera"
        />
        <div class="flex items-center">
          <USwitch v-model="isAutoMode" size="sm" />
          <span class="text-xs text-gray-500 ml-2">Auto</span>
        </div>
      </div>
      <div class="flex items-center justify-between">
        <span class="text-sm text-gray-500">Status</span>
        <UBadge v-if="isConnected" color="green" size="sm">WS Connected</UBadge>
        <UBadge v-else color="red" size="sm">WS Disconnected</UBadge>
      </div>
    </div>

    <div class="p-4 border-b border-gray-200 dark:border-gray-800">
      <UModal title="Add Vehicle" v-model="isModalOpen">
        <UButton label="Add Vehicle" color="primary" block icon="i-lucide-plus" @click="isModalOpen = true" />
        <template #body>
          <div class="space-y-4">
            <input v-model="vehicleName" type="text" placeholder="Vehicle name (optional)" class="block w-full px-3 py-2 border rounded" />
            <input type="file" multiple ref="fileInput" accept="image/*" class="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-primary-50 file:text-primary-700 hover:file:bg-primary-100" />
            
            <div class="flex justify-end gap-2">
              <UButton color="gray" variant="soft" label="Cancel" @click="isModalOpen = false" />
              <UButton color="primary" label="Upload" @click="uploadVehicle" :loading="isUploading" />
            </div>
          </div>
        </template>
      </UModal>
    </div>

    <div class="flex-1 overflow-y-auto p-4 space-y-2">
      <div v-if="vehicleItems.length === 0" class="text-sm text-gray-500 text-center py-4">
        No vehicles added yet.
      </div>
      
      <UListbox 
        v-else
        v-model="trackedVehicle" 
        :items="vehicleItems" 
        v-model:search-term="searchTerm" 
        filter 
        class="w-full" 
      >
        <template #item-trailing="{ item }">
          <UButton 
            icon="i-lucide-trash-2" 
            color="red" 
            variant="ghost" 
            size="xs" 
            @click.stop="deleteVehicle(item.value)" 
            :loading="deleting === item.value" 
            class="ml-auto"
          />
        </template>
      </UListbox>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'

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
const searchTerm = ref('')

onMounted(() => {
  fetchVehicles()
})

const uploadVehicle = async () => {
  const success = await uploadVehicleFiles(fileInput.value?.files, vehicleName.value)
  if (success) {
    isModalOpen.value = false
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