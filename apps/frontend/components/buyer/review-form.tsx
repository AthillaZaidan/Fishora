'use client'

import { useId, useState } from 'react'
import { Star } from '@phosphor-icons/react/dist/ssr'
import { Button } from '@/components/common/button'
import { Field } from '@/components/common/field'
import { ApiError } from '@/lib/api/errors'
import { submitReview, type Review } from '@/lib/api/commerce'

const RATINGS = [1, 2, 3, 4, 5] as const
const MAX_USE = 120
const MAX_COMMENT = 2000

// ApiError's shared copy answers bidding and identification, so the statuses
// this endpoint owns get their own lines instead of a generic failure.
const BY_STATUS: Record<number, string> = {
  401: 'Sign in as a buyer to write a review.',
  403: 'Only the buyer this lot was allocated to can review it.',
  409: 'This lot has not been allocated yet. Reviews open once it is.',
  422: 'The rating must be between 1 and 5 stars.',
}

export function ReviewForm({
  lotId,
  onSubmitted,
}: {
  lotId: string
  onSubmitted?: (review: Review) => void
}) {
  const groupId = useId()
  const [actualUse, setActualUse] = useState('')
  const [suitability, setSuitability] = useState(3)
  const [substitute, setSubstitute] = useState(false)
  const [comment, setComment] = useState('')
  const [useError, setUseError] = useState('')
  const [error, setError] = useState('')
  const [saved, setSaved] = useState(false)
  const [busy, setBusy] = useState(false)

  async function submit(event: React.FormEvent) {
    event.preventDefault()
    const trimmedUse = actualUse.trim()
    if (!trimmedUse) {
      setUseError('Say what you used this fish for.')
      return
    }
    const trimmedComment = comment.trim()
    setUseError('')
    setError('')
    setSaved(false)
    setBusy(true)
    try {
      const review = await submitReview(lotId, {
        actual_use: trimmedUse,
        processing_suitability: suitability,
        substitute_acceptance: substitute,
        // An empty box is not a comment, so it is left out of the body.
        ...(trimmedComment ? { comment: trimmedComment } : {}),
      })
      setSaved(true)
      setActualUse('')
      setComment('')
      onSubmitted?.(review)
    } catch (cause) {
      if (cause instanceof ApiError) {
        setError(BY_STATUS[cause.status] ?? cause.userMessage)
      } else {
        setError('Could not send the review. Try again.')
      }
    } finally {
      setBusy(false)
    }
  }

  return (
    <form className="flex flex-col gap-4 rounded-2xl bg-bg-sunken px-5 py-5" onSubmit={submit}>
      <div>
        <h2 className="text-h3 text-ink">Write a review</h2>
        <p className="text-body-sm mt-1 text-ink-muted">
          Your experience using this fish. Other buyers can read it; it is not shown as verified
          knowledge.
        </p>
      </div>

      <Field
        label="What did you use it for?"
        placeholder="Fried whole"
        maxLength={MAX_USE}
        value={actualUse}
        onChange={(event) => {
          setActualUse(event.target.value)
          setSaved(false)
        }}
        error={useError || undefined}
      />

      <fieldset>
        {/* Stored in the existing processing_suitability field (1 to 5); the
            form now asks for it as an overall star rating. */}
        <legend className="text-label text-ink">Rating</legend>
        <div className="mt-2 flex flex-wrap gap-1">
          {RATINGS.map((value) => (
            <label
              key={value}
              aria-label={`${value} ${value === 1 ? 'star' : 'stars'}`}
              className="flex min-h-11 min-w-11 cursor-pointer items-center justify-center rounded-full has-[:focus-visible]:outline-2"
            >
              <input
                type="radio"
                className="sr-only"
                name={`${groupId}-suitability`}
                value={value}
                checked={suitability === value}
                onChange={() => {
                  setSuitability(value)
                  setSaved(false)
                }}
              />
              <Star
                size={28}
                weight={value <= suitability ? 'fill' : 'regular'}
                className={value <= suitability ? 'text-accent' : 'text-ink-faint'}
                aria-hidden
              />
            </label>
          ))}
        </div>
        <p className="text-body-sm mt-2 text-ink-muted">1 star: not satisfied. 5 stars: very satisfied.</p>
      </fieldset>

      <label className="flex min-h-11 items-center gap-3 text-body-sm text-ink">
        <input
          type="checkbox"
          className="size-5"
          checked={substitute}
          onChange={(event) => {
            setSubstitute(event.target.checked)
            setSaved(false)
          }}
        />
        Works as a substitute for another species
      </label>

      <Field
        multiline
        rows={4}
        label="Notes (optional)"
        placeholder="A short note for other buyers"
        maxLength={MAX_COMMENT}
        value={comment}
        onChange={(event) => {
          setComment(event.target.value)
          setSaved(false)
        }}
      />

      <div className="flex flex-col gap-2">
        <Button block type="submit" loading={busy}>
          Send review
        </Button>
        <p className="text-body-sm min-h-5 text-ink-muted" aria-live="polite">
          {error ? <span className="text-state-error">{error}</span> : saved ? 'Review sent.' : ' '}
        </p>
      </div>
    </form>
  )
}
