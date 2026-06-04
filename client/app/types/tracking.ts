export type DetectionMatch = [string, number]

export interface Detection {
  vehicle_id: string
  box: number[]
  matches?: DetectionMatch[]
}
