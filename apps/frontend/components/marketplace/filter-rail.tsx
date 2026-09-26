'use client'

import { FilterGroups } from '@/components/marketplace/filter-groups'
import type { MarketplaceFilters } from '@/lib/marketplace-filters'

export function FilterRail({
  filters,
  onChange,
}: {
  filters: MarketplaceFilters
  onChange: (next: MarketplaceFilters) => void
}) {
  return (
    <aside className="hidden w-[264px] shrink-0 lg:block">
      <FilterGroups filters={filters} onChange={onChange} />
    </aside>
  )
}
