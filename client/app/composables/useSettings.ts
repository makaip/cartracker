import { useState } from '#imports'

export type View = 'live' | 'database'

export function useSettings() {
  const matchThreshold = useState<number>('matchThreshold', () => 0.4)
  const statusPollInterval = useState<number>('statusPollInterval', () => 2000)
  // what the left and middle panels show
  const view = useState<View>('view', () => 'live')
  // database view: hide recorded sightings that scored below this (nothing is deleted)
  const sightingMinScore = useState<number>('sightingMinScore', () => 0)

  return {
    matchThreshold,
    statusPollInterval,
    view,
    sightingMinScore
  }
}
