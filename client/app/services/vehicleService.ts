export const vehicleService = {
  async fetchVehicles(apiUrl: string) {
    const res = await fetch(`${apiUrl}/vehicles`)
    if (!res.ok) throw new Error('Failed to fetch vehicles')
    return res.json()
  },

  async addVehicle(apiUrl: string, files: FileList | null | undefined, name?: string | null) {
    if (!files || files.length === 0) return false

    const formData = new FormData()
    for (let i = 0; i < files.length; i++) {
      const file = files.item(i)
      if (file) {
        formData.append('pictures', file)
      }
    }
    if (name) formData.append('name', name)

    const res = await fetch(`${apiUrl}/add_vehicle`, {
      method: 'POST',
      body: formData
    })
    
    if (!res.ok) throw new Error('Failed to add vehicle')
    return true
  },

  async deleteVehicle(apiUrl: string, uuid: string) {
    const formData = new FormData()
    formData.append('uuid', uuid)
    
    const res = await fetch(`${apiUrl}/delete_vehicle`, {
      method: 'POST',
      body: formData
    })
    
    if (!res.ok) throw new Error('Failed to delete vehicle')
    return true
  }
}
