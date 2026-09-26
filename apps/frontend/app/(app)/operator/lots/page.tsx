import { OperatorLots } from '@/components/operator/operator-lots'
import type { Lot } from '@/lib/api/commerce'
import { requireRole } from '@/lib/api/guard'
import { listMyLotsAsServer } from '@/lib/api/server'

export default async function OperatorLotsPage() {
  await requireRole('operator', '/operator/lots')
  let lots: Lot[] = []
  try {
    lots = await listMyLotsAsServer('mine=1')
  } catch {
    lots = []
  }
  return <OperatorLots lots={lots} />
}
