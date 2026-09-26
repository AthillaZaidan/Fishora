import { IdentifyFlow } from '@/components/operator/identify-flow'
import { requireRole } from '@/lib/api/guard'
import { confirmSpecies, declareSpecies, identifyCatch, loadKnowledge } from './actions'

export default async function OperatorPage() {
  // Every action on this page needs an operator session; ask for one up front.
  await requireRole('operator', '/operator')
  return (
    <IdentifyFlow
      identifyCatch={identifyCatch}
      confirmSpecies={confirmSpecies}
      loadKnowledge={loadKnowledge}
      declareSpecies={declareSpecies}
    />
  )
}
