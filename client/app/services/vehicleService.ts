import type { Sighting, SightingPage, Vehicle } from '../types/vehicle'

function picturesForm(files: FileList) {
  const formData = new FormData()
  for (const file of Array.from(files)) formData.append('pictures', file)
  return formData
}

async function request(url: string, init: RequestInit, failure: string) {
  const res = await fetch(url, init)
  if (!res.ok) throw new Error(failure)
  return res.json()
}

export const vehicleService = {
  async fetchVehicles(apiUrl: string): Promise<Vehicle[]> {
    return request(`${apiUrl}/vehicles`, {}, 'Failed to fetch vehicles')
  },

  /** returns the uuid of the new vehicle */
  async addVehicle(apiUrl: string, files: FileList, name?: string | null): Promise<string> {
    const formData = picturesForm(files)
    if (name) formData.append('name', name)

    const created = await request(`${apiUrl}/add_vehicle`, { method: 'POST', body: formData }, 'Failed to add vehicle')
    return created.uuid
  },

  async deleteVehicle(apiUrl: string, uuid: string) {
    const formData = new FormData()
    formData.append('uuid', uuid)

    await request(`${apiUrl}/delete_vehicle`, { method: 'POST', body: formData }, 'Failed to delete vehicle')
    return true
  },

  async updateVehicle(apiUrl: string, uuid: string, patch: Partial<Pick<Vehicle, 'name' | 'tracking' | 'threshold'>>) {
    await request(`${apiUrl}/vehicles/${uuid}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patch)
    }, 'Failed to update vehicle')
  },

  /** both image calls return the vehicle's image list after the change */
  async addImages(apiUrl: string, uuid: string, files: FileList): Promise<string[]> {
    const res = await request(`${apiUrl}/vehicles/${uuid}/images`, { method: 'POST', body: picturesForm(files) }, 'Failed to add images')
    return res.images
  },

  async deleteImage(apiUrl: string, uuid: string, filename: string): Promise<string[]> {
    const res = await request(`${apiUrl}/vehicles/${uuid}/images/${encodeURIComponent(filename)}`, { method: 'DELETE' }, 'Failed to delete image')
    return res.images
  },

  imageUrl(apiUrl: string, uuid: string, filename: string) {
    return `${apiUrl}/vehicles/${uuid}/images/${encodeURIComponent(filename)}`
  },

  async fetchSightings(apiUrl: string, uuid: string, limit: number, minScore = 0): Promise<SightingPage> {
    const query = new URLSearchParams({ vehicle: uuid, limit: String(limit), min_score: String(minScore) })
    return request(`${apiUrl}/sightings?${query}`, {}, 'Failed to fetch sightings')
  },

  async deleteSighting(apiUrl: string, id: number) {
    await request(`${apiUrl}/sightings/${id}`, { method: 'DELETE' }, 'Failed to delete sighting')
  },

  async clearSightings(apiUrl: string, uuid: string) {
    await request(`${apiUrl}/sightings?vehicle=${uuid}`, { method: 'DELETE' }, 'Failed to clear sightings')
  },

  // the picture of an ongoing sighting is replaced when a better look comes in; the score changes with it
  sightingImageUrl(apiUrl: string, sighting: Sighting) {
    return `${apiUrl}/sightings/${sighting.id}/image?s=${sighting.score}`
  }
}
