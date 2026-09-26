import { PreferenceForm } from '@/components/buyer/preference-form'
import { requireRole } from '@/lib/api/guard'
import { loadSavedProfile } from './load'

// Data in, no function props: the form owns its own save and preview, so this
// stays a Server Component. Passing a closure across the boundary breaks prerender.
export default async function PreferencesPage() {
  const me = await requireRole('buyer', '/preferences')
  const { preferences, counts, loadFailed } = await loadSavedProfile(me.id)

  return (
    <div className="mx-auto max-w-[640px]">
      <header className="mb-8">
        <h1 className="text-h1 text-ink">Buying profile</h1>
        <p className="text-body-sm mt-1 max-w-[52ch] text-ink-muted">
          Tell us where you are and what you buy. Lots that fit are marked &ldquo;Matched for
          you&rdquo; in the marketplace, with the reasons on each lot.
        </p>
      </header>
      <PreferenceForm
        buyerId={me.id}
        initial={preferences}
        initialCounts={counts}
        loadFailed={loadFailed}
      />
    </div>
  )
}
