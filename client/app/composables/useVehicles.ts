import { computed } from 'vue'
import { vehicleService } from '../services/vehicleService'
import type { Vehicle } from '../types/vehicle'

export function useVehicles() {
  const trackedVehicle = useState<string | null>('trackedVehicle', () => null)
  const appConfig = useAppConfig()
  
  const backend = (appConfig.backend as any) || {
    apiUrl: 'http://localhost:8765'
  }
  
  const vehicles = useState<Vehicle[]>('vehicles', () => [])
  const isUploading = useState('isUploadingVehicle', () => false)
  const deleting = useState<string | null>('deletingVehicleUuid', () => null)

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
      vehicles.value = await vehicleService.fetchVehicles(backend.apiUrl)
    } catch (err) {
      console.error('Failed to fetch vehicles:', err)
    }
  }

  const uploadVehicleFiles = async (files: FileList | null | undefined, name?: string | null) => {
    if (!files || files.length === 0) return false

    isUploading.value = true
    try {
      const success = await vehicleService.addVehicle(backend.apiUrl, files, name)
      if (success) {
        await fetchVehicles()
        return true
      }
      return false
    } catch (err) {
      console.error('Error adding vehicle:', err)
      return false
    } finally {
      isUploading.value = false
    }
  }

  const deleteVehicleRecord = async (uuid: string) => {
    deleting.value = uuid
    try {
      const success = await vehicleService.deleteVehicle(backend.apiUrl, uuid)
      if (success) {
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
