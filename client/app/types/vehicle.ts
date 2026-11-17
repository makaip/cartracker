export interface Vehicle {
  uuid: string
  name?: string
  /** record sightings of this vehicle while the server runs */
  tracking: boolean
  /** minimum match score that is recorded as a sighting */
  threshold: number
  /** file names of the reference images */
  images: string[]
  /** number of recorded sightings */
  sightings: number
  last_seen?: number | null
  [key: string]: any
}

export interface Sighting {
  id: number
  vehicle_uuid: string
  camera_uuid: string
  camera_name: string | null
  score: number
  /** unix time in seconds */
  seen_at: number
  image: string
}

export interface SightingPage {
  total: number
  items: Sighting[]
}
