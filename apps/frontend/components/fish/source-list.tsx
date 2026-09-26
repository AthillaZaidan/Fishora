import type { SourceMetadata } from '@/lib/api/fish'
import { normaliseDashes } from '@/lib/format'

const VISIBLE = 3

// Numbered titles, collapsed past three into a disclosure.
export interface SourceListProps {
  sources: SourceMetadata[]
}

export function SourceList({ sources }: SourceListProps) {
  if (sources.length === 0) return null

  const shown = sources.slice(0, VISIBLE)
  const rest = sources.slice(VISIBLE)

  return (
    <div>
      <TitleList sources={shown} start={1} />
      {rest.length > 0 && (
        <details className="mt-2">
          <summary className="text-body-sm flex min-h-11 cursor-pointer items-center text-ink-muted">
            {rest.length} more {rest.length === 1 ? 'source' : 'sources'}
          </summary>
          <TitleList sources={rest} start={VISIBLE + 1} />
        </details>
      )}
    </div>
  )
}

function TitleList({
  sources,
  start,
}: {
  sources: SourceMetadata[]
  start: number
}) {
  return (
    <ol start={start} className="text-body-sm list-decimal space-y-1 pl-5 text-ink-muted">
      {sources.map((source) => (
        <li key={source.source_id}>{normaliseDashes(source.title)}</li>
      ))}
    </ol>
  )
}
