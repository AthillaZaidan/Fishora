'use client'

import { useEffect, useRef, useState, useSyncExternalStore } from 'react'
import { useRouter } from 'next/navigation'
import { Camera, UploadSimple, X } from '@phosphor-icons/react/dist/ssr'
import { Skeleton } from '@/components/common/skeleton'
import { Button } from '@/components/common/button'
import { Field } from '@/components/common/field'
import { Select } from '@/components/common/select'
import { FALLBACK_LANDING_POINTS } from '@/lib/landing-points'
import { KnowledgeCardView } from '@/components/fish/knowledge-card'
import { PredictionCard } from '@/components/fish/prediction-card'
import type { ActionFailure, ActionResult } from '@/lib/api/action-result'
import { listLandingPoints, publishLot as defaultPublishLot, type LandingPoint, type Lot } from '@/lib/api/commerce'
import { ApiError, messageFor } from '@/lib/api/errors'
import type {
  IdentificationResult,
  KnowledgeResponse,
  KnowledgeResult,
  ManualEntryResult,
} from '@/lib/api/fish'
import { hasCard } from '@/lib/api/fish'
import { downscaleImage } from '@/lib/image'
import { SPECIES, SUPPORTED_LABELS, type SpeciesLabel } from '@/lib/species'
import { Z } from '@/lib/z'

const DRAFT_KEY = 'fishora.operator.draft'
// The card job runs after verification. Poll until it finishes: about 90 s.
const CARD_POLL_MS = 2000
const CARD_POLL_ATTEMPTS = 45

type PublishLotPayload = {
  prediction_id: string
  quantity_kg: string
  lot_count: number
  starting_price_per_kg: string
  size_category: 'S' | 'M' | 'L'
  landing_point_id: string
  auction_minutes?: number
}

// Mirrors AUCTION_MINUTE_OPTIONS in apps/main_api/services/lots.py.
const DURATIONS = [
  { id: '30m', label: '30 min', minutes: 30 },
  { id: '1h', label: '1 hour', minutes: 60 },
  { id: '2h', label: '2 hours', minutes: 120 },
  { id: '3h', label: '3 hours', minutes: 180 },
] as const
// Mirrors MAX_LOT_COUNT in the same file.
const MAX_LOT_COUNT = 50
const SIZES = ['S', 'M', 'L'] as const

// Action-bar buttons share the row: each one may shrink below its label width
// rather than pushing the row past the viewport, and clips instead of spilling
// its pill. Padding opens up once there is room for it.
const FLEX_ACTION = 'min-w-0 flex-1 overflow-hidden px-4 sm:px-6'

function subscribeToConnectivity(onChange: () => void) {
  window.addEventListener('online', onChange)
  window.addEventListener('offline', onChange)
  return () => {
    window.removeEventListener('online', onChange)
    window.removeEventListener('offline', onChange)
  }
}
const isOnline = () => navigator.onLine
const assumeOnline = () => true

type Size = (typeof SIZES)[number]
type Step = 1 | 2 | 3 | 4

type Draft = {
  quantityKg: string
  lotCount: string
  size: Size
  pricePerKg: string
  landingPoint: string
  duration: (typeof DURATIONS)[number]['id']
}

function readDraft(): Partial<Draft> | null {
  try {
    const raw = sessionStorage.getItem(DRAFT_KEY)
    return raw ? (JSON.parse(raw) as Partial<Draft>) : null
  } catch {
    // Blocked storage or a corrupt draft: start empty.
    return null
  }
}

type FormErrors = Partial<Record<'quantity' | 'lotCount' | 'price', string>>

/** The same bounds the API enforces, checked before the request so the message sits on the field. */
function validateLot(quantityKg: string, lotCount: string, pricePerKg: string): FormErrors {
  const errors: FormErrors = {}
  const quantity = Number(quantityKg.replace(',', '.'))
  if (!quantityKg.trim() || !Number.isFinite(quantity) || quantity <= 0) {
    errors.quantity = 'Enter the weight of each lot in kg.'
  }
  const count = Number(lotCount)
  if (!Number.isInteger(count) || count < 1 || count > MAX_LOT_COUNT) {
    errors.lotCount = `Enter a number of lots from 1 to ${MAX_LOT_COUNT}.`
  }
  const price = Number(pricePerKg)
  if (!pricePerKg.trim() || !Number.isFinite(price) || price <= 0) {
    errors.price = 'Enter a starting price per kg.'
  }
  return errors
}

export interface IdentifyFlowProps {
  identifyCatch: (formData: FormData) => Promise<ActionResult<IdentificationResult>>
  confirmSpecies: (
    predictionId: string,
    verifiedSpeciesId: string
  ) => Promise<
    ActionResult<{
      prediction_id: string
      predicted_species_id: string
      verified_species_id: string
      verification_status: 'confirmed' | 'corrected'
    }>
  >
  loadKnowledge: (predictionId: string) => Promise<ActionResult<KnowledgeResult>>
  declareSpecies: (
    formData: FormData,
    speciesId: string
  ) => Promise<ActionResult<ManualEntryResult>>
  publishLot?: (payload: PublishLotPayload) => Promise<Lot | unknown>
}

// Four steps, forward-only. A 503 or 502 never blocks the operator.
export function IdentifyFlow({
  identifyCatch,
  confirmSpecies,
  loadKnowledge,
  declareSpecies,
  publishLot = defaultPublishLot,
}: IdentifyFlowProps) {
  const router = useRouter()
  const cameraRef = useRef<HTMLInputElement>(null)
  const uploadRef = useRef<HTMLInputElement>(null)
  const previewRef = useRef<string | null>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const [live, setLive] = useState(false)

  const [step, setStep] = useState<Step>(1)
  // Connectivity is external state. Branching on `typeof navigator` in the
  // initialiser is the server/client branch React's hydration warning names,
  // and it made the offline banner differ between the two renders.
  const online = useSyncExternalStore(subscribeToConnectivity, isOnline, assumeOnline)
  const [image, setImage] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [identifyError, setIdentifyError] = useState<ActionFailure | null>(null)
  const [manualOpen, setManualOpen] = useState(false)
  const [prediction, setPrediction] = useState<IdentificationResult | null>(null)
  const [knowledge, setKnowledge] = useState<KnowledgeResponse | null>(null)
  const [knowledgeStatus, setKnowledgeStatus] = useState<'idle' | 'loading' | 'ready' | 'unavailable'>('idle')
  const [knowledgeMessage, setKnowledgeMessage] = useState('')
  // The prediction whose card is being awaited; cleared when the operator
  // moves on, so a late reply never lands on the next catch.
  const cardPoll = useRef<string | null>(null)
  const [label, setLabel] = useState<string>('tenggiri')
  const [quantityKg, setQuantityKg] = useState('')
  const [lotCount, setLotCount] = useState('1')
  const [size, setSize] = useState<Size>('M')
  const [pricePerKg, setPricePerKg] = useState('')
  const [landingPoints, setLandingPoints] = useState<LandingPoint[]>(FALLBACK_LANDING_POINTS)
  const [landingPoint, setLandingPoint] = useState<string>(FALLBACK_LANDING_POINTS[0].id)
  const [duration, setDuration] = useState<(typeof DURATIONS)[number]['id']>('1h')
  const [publishError, setPublishError] = useState('')
  const [formErrors, setFormErrors] = useState<FormErrors>({})
  const restored = useRef(false)

  // The API's list is the one publish validates against; the static list is
  // only a fallback so the form still works when the request fails.
  useEffect(() => {
    let cancelled = false
    listLandingPoints()
      .then((points) => {
        if (cancelled || points.length === 0) return
        setLandingPoints(points)
        setLandingPoint((current) => (points.some((point) => point.id === current) ? current : points[0].id))
      })
      .catch(() => {
        // Keep the fallback list.
      })
    return () => {
      cancelled = true
    }
  }, [])

  // Restore the lot details typed before a reload or a dropped connection.
  // The photo and the identification cannot be kept (a File does not survive
  // storage), so the flow restarts at the photo with the form filled in.
  useEffect(() => {
    const draft = readDraft()
    restored.current = true
    if (!draft) return
    // Deferred: a one-off sync from external storage, not render-driven state.
    queueMicrotask(() => {
      if (typeof draft.quantityKg === 'string') setQuantityKg(draft.quantityKg)
      if (typeof draft.lotCount === 'string') setLotCount(draft.lotCount)
      if (draft.size && SIZES.includes(draft.size)) setSize(draft.size)
      if (typeof draft.pricePerKg === 'string') setPricePerKg(draft.pricePerKg)
      if (typeof draft.landingPoint === 'string') setLandingPoint(draft.landingPoint)
      if (draft.duration && DURATIONS.some((option) => option.id === draft.duration)) setDuration(draft.duration)
    })
  }, [])

  useEffect(() => {
    // Not before the restore has read it: the first render would overwrite it.
    if (!restored.current) return
    try {
      sessionStorage.setItem(
      DRAFT_KEY,
      JSON.stringify({
        quantityKg,
        lotCount,
        size,
        pricePerKg,
        landingPoint,
        duration,
        imageName: image?.name ?? null,
        step,
      })
    )
    } catch {
      // Storage full or blocked: the form still works, only unsaved.
    }
  }, [quantityKg, lotCount, size, pricePerKg, landingPoint, duration, image, step])

  useEffect(() => {
    return () => {
      if (previewRef.current) URL.revokeObjectURL(previewRef.current)
      cardPoll.current = null
      // A live track keeps the camera indicator lit after the operator leaves.
      streamRef.current?.getTracks().forEach((track) => track.stop())
    }
  }, [])

  function stopCamera() {
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
    setLive(false)
  }

  async function startCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        // The rear camera is the one pointed at the catch.
        video: { facingMode: { ideal: 'environment' } },
        audio: false,
      })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }
      setLive(true)
    } catch {
      // No camera, permission refused, or a page served over plain http from a
      // LAN address, where getUserMedia is unavailable. The file input still
      // reaches the phone's own camera app, so the step is never a dead end.
      cameraRef.current?.click()
    }
  }

  async function shoot() {
    const video = videoRef.current
    if (!video || !video.videoWidth) return
    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    canvas.getContext('2d')?.drawImage(video, 0, 0)
    const blob = await new Promise<Blob | null>((resolve) =>
      canvas.toBlob(resolve, 'image/jpeg', 0.9)
    )
    if (!blob) return
    stopCamera()
    takeFile(new File([blob], 'tangkapan.jpg', { type: 'image/jpeg' }))
  }

  function takeFile(file: File | undefined) {
    if (!file) return
    if (previewRef.current) URL.revokeObjectURL(previewRef.current)
    const url = URL.createObjectURL(file)
    previewRef.current = url
    setImage(file)
    setPreview(url)
    setIdentifyError(null)
    setManualOpen(false)
  }

  async function runIdentify() {
    if (!image) return
    setBusy(true)
    setIdentifyError(null)
    // Decode and upload are reported separately. Folding them together tells an
    // operator to retake the photo when the real problem is that the service is
    // unreachable, which sends them round a loop that cannot succeed.
    let scaled: Blob
    try {
      scaled = await downscaleImage(image)
    } catch {
      setIdentifyError({
        ok: false,
        kind: 'image_invalid',
        userMessage: 'Unsupported image format. Use JPG or PNG.',
        retryable: true,
        status: 0,
      })
      setBusy(false)
      return
    }

    try {
      const body = new FormData()
      body.append('file', scaled)
      const result = await identifyCatch(body)
      if (!result.ok) {
        setIdentifyError(result)
        return
      }
      setPrediction(result.data)
      setLabel(result.data.prediction.normalized_label)
      setStep(2)
    } catch {
      // A transport failure the action could not classify. Without this the
      // operator taps Identify and nothing happens at all: no advance, no
      // error, no way to know why.
      setIdentifyError({
        ok: false,
        kind: 'offline',
        userMessage: messageFor('offline'),
        retryable: true,
        status: 0,
      })
    } finally {
      setBusy(false)
    }
  }

  async function runConfirm(speciesId: string) {
    if (!prediction) return
    setBusy(true)
    setIdentifyError(null)
    try {
      const verified = await confirmSpecies(prediction.prediction_id, speciesId)
      if (!verified.ok) {
        // Shown on this step (below the card), not only on the photo step.
        setIdentifyError(verified)
        return
      }
      setStep(3)
      void fetchCard(prediction.prediction_id)
    } catch {
      setIdentifyError({ ok: false, kind: 'offline', userMessage: messageFor('offline'), retryable: true, status: 0 })
    } finally {
      setBusy(false)
    }
  }

  // "Species not listed": back to the photo, with the full species list open.
  // The photo is kept, so the operator either names the fish or retakes it.
  function speciesNotListed() {
    setPrediction(null)
    setIdentifyError(null)
    setManualOpen(true)
    setStep(1)
  }

  async function pickManual(species: SpeciesLabel) {
    if (!image) return
    setManualOpen(false)
    setBusy(true)
    try {
      const body = new FormData()
      body.append('file', await downscaleImage(image))
      const declared = await declareSpecies(body, `species_${species}`)
      if (!declared.ok) {
        setIdentifyError(declared)
        return
      }
      setPrediction({
        prediction_id: declared.data.prediction_id,
        model_version: declared.data.model_version,
        status: 'confident_prediction',
        prediction: {
          species_id: declared.data.verified_species_id,
          normalized_label: declared.data.normalized_label,
          confidence: 0,
        },
        top_candidates: [],
        threshold: 0,
        verification_status: 'pending',
      })
      setLabel(species)
      setIdentifyError(null)

      setStep(3)
      void fetchCard(declared.data.prediction_id)
    } finally {
      setBusy(false)
    }
  }

  // A 202 means the card job is still running; ask again until it has a card,
  // fails, or runs out of time. The operator can publish meanwhile.
  async function fetchCard(predictionId: string) {
    cardPoll.current = predictionId
    setKnowledge(null)
    setKnowledgeMessage('')
    setKnowledgeStatus('loading')
    for (let attempt = 0; attempt < CARD_POLL_ATTEMPTS; attempt++) {
      const result = await loadKnowledge(predictionId)
      if (cardPoll.current !== predictionId) return
      if (result.ok && hasCard(result.data)) {
        setKnowledge(result.data)
        setKnowledgeStatus('ready')
        return
      }
      if (!result.ok) {
        setKnowledgeMessage(result.userMessage)
        setKnowledgeStatus('unavailable')
        return
      }
      await new Promise((resolve) => setTimeout(resolve, CARD_POLL_MS))
      if (cardPoll.current !== predictionId) return
    }
    setKnowledgeMessage('The knowledge card is not ready yet.')
    setKnowledgeStatus('unavailable')
  }

  async function runPublish() {
    if (!prediction) return
    const errors = validateLot(quantityKg, lotCount, pricePerKg)
    setFormErrors(errors)
    if (Object.keys(errors).length > 0) return
    setBusy(true)
    setPublishError('')
    try {
      await publishLot({
        prediction_id: prediction.prediction_id,
        quantity_kg: quantityKg,
        lot_count: Number(lotCount),
        starting_price_per_kg: pricePerKg,
        size_category: size,
        landing_point_id: landingPoint,
        auction_minutes: DURATIONS.find((option) => option.id === duration)?.minutes ?? 60,
      })
      try {
        sessionStorage.removeItem(DRAFT_KEY)
      } catch {
        // Nothing to clear.
      }
      router.push('/operator/lots')
    } catch (cause) {
      setPublishError(
        cause instanceof ApiError && cause.status === 422
          ? 'Check the lot details: a value was refused.'
          : cause instanceof ApiError && cause.status === 409
            ? 'This catch has already been published.'
            : cause instanceof ApiError
              ? cause.userMessage
              : 'Something went wrong. Try again.'
      )
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="flex h-[calc(100dvh-7rem)] flex-col lg:h-[calc(100dvh-3.5rem)]">
      <div className="mx-auto flex w-full max-w-lg flex-1 flex-col overflow-y-auto pt-4">
        <p className="text-num-sm text-ink-muted">Step {step} of 4</p>
        <div className="mt-2 flex gap-1" aria-hidden>
          {([1, 2, 3, 4] as const).map((n) => (
            <span
              key={n}
              className={[
                'h-px flex-1',
                n <= step ? 'bg-ink' : 'bg-line',
              ].join(' ')}
            />
          ))}
        </div>

        {!online && (
          <p className="text-body-sm mt-4 rounded-[var(--radius-input)] border border-state-warn px-3 py-3 text-state-warn">
            No connection. The lot details you have typed are kept on this device.
          </p>
        )}

        {step === 1 && (
          <CaptureStep
            live={live}
            videoRef={videoRef}
            preview={preview}
            error={identifyError}
            manualOpen={manualOpen}
            onManualOpen={() => setManualOpen(true)}
            onPickManual={pickManual}
            onRetry={runIdentify}
          />
        )}

        {step === 2 && prediction && (
          <div className="mt-6">
            <PredictionCard result={prediction} onConfirm={runConfirm} onSpeciesNotListed={speciesNotListed} />
            {identifyError && (
              <p className="text-body-sm mt-4 rounded-[var(--radius-input)] border border-state-error px-3 py-3 text-state-error">
                {identifyError.userMessage}
              </p>
            )}
          </div>
        )}

        {step === 3 && (
          <div className="mt-6 flex min-h-0 flex-1 flex-col gap-4">
            {knowledgeStatus === 'loading' && (
              <div className="flex flex-col gap-3" aria-live="polite">
                <p className="text-body-sm text-ink-muted">Preparing the knowledge card… You can publish the lot meanwhile.</p>
                <Skeleton className="h-10 w-full" />
                <Skeleton className="h-10 w-full" />
                <Skeleton className="h-10 w-2/3" />
              </div>
            )}
            {knowledgeStatus === 'unavailable' && prediction && (
              <div className="flex flex-col gap-2 rounded-[var(--radius-input)] border border-state-warn px-3 py-3">
                <p className="text-body-sm text-state-warn">
                  {knowledgeMessage} You can still publish the lot.
                </p>
                <Button variant="secondary" size="sm" onClick={() => void fetchCard(prediction.prediction_id)}>
                  Try again
                </Button>
              </div>
            )}
            {knowledgeStatus === 'ready' && knowledge?.card && (
              <KnowledgeCardView card={knowledge.card} label={label} />
            )}
          </div>
        )}

        {step === 4 && (
          <>
            <LotForm
              label={label}
              quantityKg={quantityKg}
              lotCount={lotCount}
              onLotCount={setLotCount}
              size={size}
              pricePerKg={pricePerKg}
              landingPoint={landingPoint}
              landingPoints={landingPoints}
              errors={formErrors}
              duration={duration}
              onQuantity={setQuantityKg}
              onSize={setSize}
              onPrice={setPricePerKg}
              onLanding={setLandingPoint}
              onDuration={setDuration}
            />
            {publishError && (
              <p className="text-body-sm mt-4 text-state-error">{publishError}</p>
            )}
          </>
        )}
      </div>

      <div className="shrink-0" style={{ zIndex: Z.actionBar }}>
        <div className="mx-auto flex w-full max-w-lg items-center gap-2 px-4 py-3 pb-[calc(0.75rem+env(safe-area-inset-bottom))] lg:px-0">
        <input
          ref={cameraRef}
          type="file"
          accept="image/*"
          capture="environment"
          className="sr-only"
          tabIndex={-1}
          aria-hidden
          onChange={(event) => takeFile(event.target.files?.[0])}
        />
        <input
          ref={uploadRef}
          type="file"
          accept="image/*"
          className="sr-only"
          tabIndex={-1}
          aria-hidden
          onChange={(event) => takeFile(event.target.files?.[0])}
        />
        {step === 1 && !image && (
          <>
            {live ? (
              <Button size="lg" className={FLEX_ACTION} onClick={shoot}>
                Take photo
              </Button>
            ) : (
              <Button
                size="lg"
                className={FLEX_ACTION}
                icon={<Camera size={20} />}
                onClick={startCamera}
              >
                Camera
              </Button>
            )}
            <IconButton
              label={live ? 'Close camera' : 'Upload a file'}
              onClick={live ? stopCamera : () => uploadRef.current?.click()}
            >
              {live ? <X size={20} /> : <UploadSimple size={20} />}
            </IconButton>
          </>
        )}
        {step === 1 && image && (
          <>
            <Button
              size="lg"
              variant="secondary"
              className={FLEX_ACTION}
              onClick={() => {
                setImage(null)
                setPreview(null)
                setIdentifyError(null)
              }}
            >
              Retake
            </Button>
            <Button size="lg" className={FLEX_ACTION} loading={busy} onClick={runIdentify}>
              Identify
            </Button>
          </>
        )}
        {step === 3 && (
          <Button size="lg" className={FLEX_ACTION} onClick={() => setStep(4)}>
            Continue
          </Button>
        )}
        {step === 4 && (
          <Button
            size="lg"
            className={FLEX_ACTION}
            type="button"
            loading={busy}
            onClick={runPublish}
          >
            Publish
          </Button>
        )}
        </div>
      </div>
    </div>
  )
}

function IconButton({
  label,
  onClick,
  children,
}: {
  label: string
  onClick: () => void
  children: React.ReactNode
}) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      onClick={onClick}
      className="grid size-13 shrink-0 place-items-center rounded-full border border-line-strong text-ink transition-colors hover:bg-bg-sunken active:scale-[0.98]"
    >
      {children}
    </button>
  )
}

function CaptureStep({
  live,
  videoRef,
  preview,
  error,
  manualOpen,
  onManualOpen,
  onPickManual,
  onRetry,
}: {
  live: boolean
  videoRef: React.RefObject<HTMLVideoElement | null>
  preview: string | null
  error: ActionFailure | null
  manualOpen: boolean
  onManualOpen: () => void
  onPickManual: (label: SpeciesLabel) => void
  onRetry: () => void
}) {
  return (
    <div className="mt-6 flex min-h-0 flex-1 flex-col gap-4">
      <p className="text-body-sm text-ink-muted">
        Whole fish, plain background, good light.
      </p>
      {/* One frame for all three states, so the layout never jumps between
          them. The video stays mounted because its ref has to exist before the
          stream can be attached to it. */}
      {/* Fills the space the column has left rather than forcing a fixed
          aspect, with a floor so it stays usable on a short window. */}
      <div className="relative min-h-52 w-full flex-1 overflow-hidden rounded-2xl">
        <video
          ref={videoRef}
          playsInline
          muted
          className={`size-full object-cover ${live ? '' : 'hidden'}`}
        />

        {!live &&
          (preview ? (
            // eslint-disable-next-line @next/next/no-img-element -- blob URL from the capture
            <img src={preview} alt="Your catch" className="size-full object-cover" />
          ) : (
            <div
              className="text-body-sm flex size-full flex-col items-center justify-center gap-2 rounded-2xl border border-dashed border-line-strong text-ink-muted"
              aria-hidden
            >
              <Camera size={28} />
              <p>No photo yet</p>
            </div>
          ))}
      </div>
      {error && (
        <div className="flex flex-col gap-3 rounded-[var(--radius-input)] border border-state-error px-3 py-3">
          <p className="text-body-sm text-state-error">{error.userMessage}</p>
          {(error.kind === 'photo_not_fish' || error.kind === 'photo_unknown_species') && (
            // Retaking is the bar's own action; naming the species by hand is
            // the way on when the model cannot place the fish.
            <Button size="lg" variant="secondary" block onClick={onManualOpen}>
              Choose the species myself
            </Button>
          )}
          {error.kind === 'cv_unavailable' && (
            <div className="flex flex-col gap-2">
              <Button size="lg" block onClick={onRetry}>
                Try again
              </Button>
              <Button size="lg" variant="secondary" block onClick={onManualOpen}>
                Choose the species myself
              </Button>
            </div>
          )}
        </div>
      )}
      {manualOpen && (
        <ul className="flex flex-col gap-2">
          {SUPPORTED_LABELS.map((species) => (
            <li key={species}>
              <Button
                size="lg"
                variant="secondary"
                block
                onClick={() => onPickManual(species)}
              >
                {SPECIES[species].commonName}
              </Button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

function LotForm({
  label,
  quantityKg,
  lotCount,
  onLotCount,
  size,
  pricePerKg,
  landingPoint,
  landingPoints,
  errors,
  duration,
  onQuantity,
  onSize,
  onPrice,
  onLanding,
  onDuration,
}: {
  label: string
  quantityKg: string
  lotCount: string
  onLotCount: (value: string) => void
  size: Size
  pricePerKg: string
  landingPoint: string
  landingPoints: LandingPoint[]
  errors: FormErrors
  duration: string
  onQuantity: (value: string) => void
  onSize: (value: Size) => void
  onPrice: (value: string) => void
  onLanding: (value: string) => void
  onDuration: (value: (typeof DURATIONS)[number]['id']) => void
}) {
  const resolved = SPECIES[label as SpeciesLabel]
  const total = Number(quantityKg) * Number(lotCount)
  return (
    <form className="mt-6 flex flex-col gap-5" onSubmit={(event) => event.preventDefault()}>
      <div className="rounded-[var(--radius-input)] bg-bg-sunken px-3 py-3">
        <p className="text-h3 text-ink">{resolved?.commonName ?? label}</p>
        <p className="text-body-sm text-ink-muted">
          {landingPoints.find((point) => point.id === landingPoint)?.name ?? landingPoint}
        </p>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <Field
          label="Weight per lot"
          inputMode="decimal"
          value={quantityKg}
          onChange={(event) => onQuantity(event.target.value)}
          suffix="kg"
          helper="The weight of each lot."
          error={errors.quantity}
        />
        <Field
          label="Number of lots"
          inputMode="numeric"
          value={lotCount}
          onChange={(event) => onLotCount(event.target.value.replace(/\D/g, ''))}
          suffix="lots"
          helper={`1 to ${MAX_LOT_COUNT}. Each lot is auctioned separately.`}
          error={errors.lotCount}
        />
      </div>
      {total > 0 && Number(lotCount) > 1 && (
        <p className="text-body-sm -mt-2 text-ink-muted tabular-nums">
          Total {lotCount} lots × {quantityKg} kg = {total.toLocaleString('en-GB')} kg
        </p>
      )}
      <fieldset>
        <legend className="text-label mb-2 text-ink">Size category</legend>
        <div className="grid grid-cols-3 gap-2">
          {SIZES.map((option) => (
            <button
              key={option}
              type="button"
              onClick={() => onSize(option)}
              className={[
                'min-h-12 rounded-full border text-body',
                option === size
                  ? 'border-ink bg-ink text-bg'
                  : 'border-line-input bg-surface text-ink',
              ].join(' ')}
            >
              {option}
            </button>
          ))}
        </div>
      </fieldset>
      <Field
        label="Starting price per kg"
        inputMode="numeric"
        value={pricePerKg}
        onChange={(event) => onPrice(event.target.value)}
        prefix="Rp"
        helper="The price the auction opens at."
        error={errors.price}
      />
      <Select
        label="Landing point"
        value={landingPoint}
        onChange={(event) => onLanding(event.target.value)}
        options={landingPoints.map((point) => ({ value: point.id, label: point.name }))}
      />
      <fieldset>
        <legend className="text-label mb-2 text-ink">Auction length</legend>
        <div className="grid grid-cols-4 gap-2">
          {DURATIONS.map((option) => (
            <button
              key={option.id}
              type="button"
              onClick={() => onDuration(option.id)}
              className={[
                'min-h-12 rounded-full border text-body-sm',
                option.id === duration
                  ? 'border-ink bg-ink text-bg'
                  : 'border-line-input bg-surface text-ink',
              ].join(' ')}
            >
              {option.label}
            </button>
          ))}
        </div>
      </fieldset>
    </form>
  )
}
