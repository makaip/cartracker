export interface CameraOption {
  value: string
  label: string
  disabled?: boolean
}

export type CameraMap = Record<string, string>
export type CameraStatusMap = Record<string, boolean>
export type CameraDetectionMap = Record<string, any[]>
