export const cameraService = {
  async fetchCameras(apiUrl: string) {
    const res = await fetch(`${apiUrl}/cameras`)
    if (!res.ok) throw new Error('Failed to fetch cameras')
    return res.json()
  },

  async fetchCameraStatus(apiUrl: string) {
    const res = await fetch(`${apiUrl}/camera_status`)
    if (!res.ok) throw new Error('Failed to fetch camera status')
    return res.json()
  }
}
