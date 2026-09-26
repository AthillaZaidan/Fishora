'use client'

import type { MarketplaceFilters } from '@/lib/marketplace-filters'

// Price only: market testing found buyers pick by price, and the search box
// already covers species and characteristics.
export function FilterGroups({
  filters,
  onChange,
}: {
  filters: MarketplaceFilters
  onChange: (next: MarketplaceFilters) => void
}) {
  return (
    <div className="flex flex-col">
      <fieldset className="border-t border-line py-4">
        <legend className="text-label text-ink">Harga per kg</legend>
        <div className="mt-3 flex gap-2">
          <input
            aria-label="Harga minimum"
            inputMode="numeric"
            placeholder="Min"
            className="min-h-11 w-full rounded-[var(--radius-input)] border border-line-input px-3 text-ink"
            value={filters.minPrice}
            onChange={(event) => onChange({ ...filters, minPrice: event.target.value })}
          />
          <input
            aria-label="Harga maksimum"
            inputMode="numeric"
            placeholder="Maks"
            className="min-h-11 w-full rounded-[var(--radius-input)] border border-line-input px-3 text-ink"
            value={filters.maxPrice}
            onChange={(event) => onChange({ ...filters, maxPrice: event.target.value })}
          />
        </div>
      </fieldset>
    </div>
  )
}
