import { useState } from '#imports'

export function useSettings() {
  const matchThreshold = useState<number>('matchThreshold', () => 0.4)
  const statusPollInterval = useState<number>('statusPollInterval', () => 2000)

  return {
    matchThreshold,
    statusPollInterval
  }
}
