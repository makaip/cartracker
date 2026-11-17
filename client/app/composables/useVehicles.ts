import { computed } from 'vue'
import { vehicleService } from '../services/vehicleService'
import type { Vehicle } from '../types/vehicle'

export function vehicleLabel(v: Vehicle) {
  return (v.name && v.name.length > 0) ? v.name : v.uuid.split('-')[0]!
}

export function useVehicles() {
  const trackedVehicle = useState<string | null>('trackedVehicle', () => null)
  // vehicle open in the database tab
  const databaseVehicle = useState<string | null>('databaseVehicle', () => null)
  const appConfig = useAppConfig()

  // vehicles and sightings are shared by all nodes, so the first one answers for everything
  const apiUrl: string = (appConfig.backend as any)?.apiUrl ?? 'http://localhost:8765'

  const vehicles = useState<Vehicle[]>('vehicles', () => [])
  const isUploading = useState('isUploadingVehicle', () => false)
  const deleting = useState<string | null>('deletingVehicleUuid', () => null)

  const vehicleItems = computed(() => {
    return vehicles.value.map(v => ({
      label: vehicleLabel(v),
      description: v.uuid,
      value: v.uuid,
      icon: 'i-lucide-car'
    }))
  })

  const fetchVehicles = async () => {
    try {
      vehicles.value = await vehicleService.fetchVehicles(apiUrl)
    } catch (err) {
      console.error('Failed to fetch vehicles:', err)
    }
  }

  /** returns the new vehicle's uuid, or null if nothing was added */
  const uploadVehicleFiles = async (files: FileList | null | undefined, name?: string | null) => {
    if (!files || files.length === 0) return null

    isUploading.value = true
    try {
      const uuid = await vehicleService.addVehicle(apiUrl, files, name)
      await fetchVehicles()
      return uuid
    } catch (err) {
      console.error('Error adding vehicle:', err)
      return null
    } finally {
      isUploading.value = false
    }
  }

  const deleteVehicleRecord = async (uuid: string) => {
    deleting.value = uuid
    try {
      await vehicleService.deleteVehicle(apiUrl, uuid)
      if (trackedVehicle.value === uuid) trackedVehicle.value = null
      if (databaseVehicle.value === uuid) databaseVehicle.value = null
      await fetchVehicles()
      return true
    } catch (err) {
      console.error('Error deleting vehicle:', err)
      return false
    } finally {
      deleting.value = null
    }
  }

  const patchLocal = (uuid: string, patch: Partial<Vehicle>) => {
    vehicles.value = vehicles.value.map(v => v.uuid === uuid ? { ...v, ...patch } : v)
  }

  const updateVehicle = async (uuid: string, patch: Partial<Pick<Vehicle, 'name' | 'tracking' | 'threshold'>>) => {
    patchLocal(uuid, patch) // show the change right away
    try {
      await vehicleService.updateVehicle(apiUrl, uuid, patch)
      return true
    } catch (err) {
      console.error('Error updating vehicle:', err)
      await fetchVehicles() // put back what the server has
      return false
    }
  }

  const addVehicleImages = async (uuid: string, files: FileList | null | undefined) => {
    if (!files || files.length === 0) return false
    try {
      patchLocal(uuid, { images: await vehicleService.addImages(apiUrl, uuid, files) })
      return true
    } catch (err) {
      console.error('Error adding images:', err)
      return false
    }
  }

  const deleteVehicleImage = async (uuid: string, filename: string) => {
    try {
      patchLocal(uuid, { images: await vehicleService.deleteImage(apiUrl, uuid, filename) })
      return true
    } catch (err) {
      console.error('Error deleting image:', err)
      return false
    }
  }

  return {
    apiUrl,
    trackedVehicle,
    databaseVehicle,
    vehicles,
    vehicleItems,
    isUploading,
    deleting,
    fetchVehicles,
    uploadVehicleFiles,
    deleteVehicleRecord,
    updateVehicle,
    addVehicleImages,
    deleteVehicleImage
  }
}
