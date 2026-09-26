'use client'

import { Button } from '@/components/common/button'
import { Sheet } from '@/components/common/sheet'
import { FilterGroups } from '@/components/marketplace/filter-groups'
import type { MarketplaceFilters } from '@/lib/marketplace-filters'

export function FilterSheet({
  open,
  onClose,
  filters,
  onChange,
  resultCount,
}: {
  open: boolean
  onClose: () => void
  filters: MarketplaceFilters
  onChange: (next: MarketplaceFilters) => void
  resultCount: number
}) {
  return (
    <Sheet
      open={open}
      onClose={onClose}
      title="Filter"
      footer={
        <Button block type="button" onClick={onClose}>
          Show {resultCount} {resultCount === 1 ? 'lot' : 'lots'}
        </Button>
      }
    >
      <FilterGroups filters={filters} onChange={onChange} />
    </Sheet>
  )
}
