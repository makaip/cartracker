import { ref, computed } from 'vue'

export function useVehicles() {
  const trackedVehicle = useState<string | null>('trackedVehicle', () => null)
  const vehicles = ref<any[]>([])
  const isUploading = ref(false)
  const deleting = ref<string | null>(null)

  const vehicleItems = computed(() => {
    return vehicles.value.map(v => ({
      label: (v.name && v.name.length > 0) ? v.name : v.uuid.split('-')[0],
      description: v.uuid,
      value: v.uuid,
      icon: 'i-lucide-car'
    }))
  })

  const fetchVehicles = async () => {
    try {
      const res = await fetch('http://localhost:8765/vehicles')
      if (res.ok) {
        vehicles.value = await res.json()
      }
    } catch (err) {
      console.error('Failed to fetch vehicles:', err)
    }
  }

  const uploadVehicleFiles = async (files: FileList | null | undefined, name?: string | null) => {
    if (!files || files.length === 0) return false

    isUploading.value = true
    const formData = new FormData()
    for (let i = 0; i < files.length; i++) {
      const file = files.item(i)
      if (file) {
        formData.append('pictures', file)
      }
    }
    if (name) formData.append('name', name)

    try {
      const res = await fetch('http://localhost:8765/add_vehicle', {
        method: 'POST',
        body: formData
      })
      
      if (res.ok) {
        await fetchVehicles()
        return true
      } else {
        console.error('Failed to add vehicle')
        return false
      }
    } catch (err) {
      console.error('Error adding vehicle:', err)
      return false
    } finally {
      isUploading.value = false
    }
  }

  const deleteVehicleRecord = async (uuid: string) => {
    deleting.value = uuid
    const formData = new FormData()
    formData.append('uuid', uuid)
    
    try {
      const res = await fetch('http://localhost:8765/delete_vehicle', {
        method: 'POST',
        body: formData
      })
      
      if (res.ok) {
        if (trackedVehicle.value === uuid) trackedVehicle.value = null
        await fetchVehicles()
        return true
      }
      return false
    } catch (err) {
      console.error('Error deleting vehicle:', err)
      return false
    } finally {
      deleting.value = null
    }
  }

  return {
    trackedVehicle,
    vehicles,
    vehicleItems,
    isUploading,
    deleting,
    fetchVehicles,
    uploadVehicleFiles,
    deleteVehicleRecord
  }
}
