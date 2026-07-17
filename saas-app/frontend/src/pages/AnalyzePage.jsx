import { useState, useRef, useCallback, useEffect } from 'react'
import Footer from '../components/Footer'

const API_BASE = 'http://localhost:8000'

/* ─── Scan Loader ────────────────────────────────────────────────────────── */
function ScanLoader() {
  return (
    <div className="flex flex-col items-center gap-6 py-12">
      <div className="relative w-32 h-32">
        <div className="absolute inset-0 rounded-full border-2 border-fresh/30 animate-pulse-ring" />
        <div className="absolute inset-4 rounded-full border-2 border-fresh/50 animate-pulse-ring" style={{ animationDelay: '0.3s' }} />
        <div className="absolute inset-0 flex items-center justify-center text-5xl animate-float">🔬</div>
        <div className="absolute inset-0 overflow-hidden rounded-full">
          <div className="scan-line" />
        </div>
      </div>
      <div className="text-center">
        <p className="text-white font-semibold text-lg">Analyse en cours…</p>
        <p className="text-forest-400 text-sm mt-1">Le modèle MobileNetV2 traite votre image</p>
      </div>
    </div>
  )
}

/* ─── Result Card ────────────────────────────────────────────────────────── */
function ResultCard({ result, preview }) {
  const isFresh = result.status === 'fresh'
  const confidence = Math.round(result.confidence * 100)

  return (
    <div className={`animate-fade-up card border-2 ${isFresh ? 'border-fresh/50' : 'border-rotten/50'}`}>
      <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4 mb-6">
        {preview && (
          <img
            src={preview}
            alt="fruit analysé"
            className="w-24 h-24 rounded-2xl object-cover border-2 border-forest-700 flex-shrink-0"
          />
        )}
        <div className="flex-1">
          <div className={`mb-2 ${isFresh ? 'badge-fresh' : 'badge-rotten'}`}>
            {isFresh ? '✅ Frais' : '⚠️ Avarié / Abîmé'}
          </div>
          <h2 className="font-display font-bold text-2xl text-white">
            {result.emoji} {result.label_fr}
          </h2>
          <div className="mt-2 flex items-center gap-3">
            <div className="flex-1 bg-forest-800 rounded-full h-2 overflow-hidden">
              <div
                className={`h-2 rounded-full transition-all duration-700 ${isFresh ? 'bg-fresh' : 'bg-rotten'}`}
                style={{ width: `${confidence}%` }}
              />
            </div>
            <span className={`text-sm font-semibold ${isFresh ? 'text-fresh' : 'text-rotten'}`}>
              {confidence}%
            </span>
          </div>
          <p className="text-forest-500 text-xs mt-1">Confiance de la prédiction</p>
        </div>
      </div>

      <div className={`rounded-2xl p-5 ${isFresh ? 'bg-fresh/10 border border-fresh/20' : 'bg-rotten/10 border border-rotten/20'}`}>
        <h3 className="font-semibold text-white mb-3">{result.suggestions.title}</h3>
        <ul className="space-y-2.5">
          {result.suggestions.tips.map((tip, i) => (
            <li key={i} className="flex gap-3 items-start text-sm text-forest-200">
              <span className={`mt-0.5 flex-shrink-0 ${isFresh ? 'text-fresh' : 'text-rotten'}`}>
                {isFresh ? '💡' : '♻️'}
              </span>
              {tip}
            </li>
          ))}
        </ul>
      </div>
    </div>
  )
}

/* ─── Camera Modal ───────────────────────────────────────────────────────── */
function CameraModal({ onCapture, onClose }) {
  const videoRef  = useRef(null)
  const streamRef = useRef(null)
  const [ready, setReady]       = useState(false)
  const [camError, setCamError] = useState(null)

  /* Start stream */
  useEffect(() => {
    let cancelled = false
    navigator.mediaDevices
      .getUserMedia({ video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } } })
      .then((stream) => {
        if (cancelled) { stream.getTracks().forEach(t => t.stop()); return }
        streamRef.current = stream
        if (videoRef.current) {
          videoRef.current.srcObject = stream
          videoRef.current.onloadedmetadata = () => setReady(true)
        }
      })
      .catch((err) => {
        if (!cancelled) setCamError(`Caméra inaccessible : ${err.message}`)
      })
    return () => {
      cancelled = true
      streamRef.current?.getTracks().forEach(t => t.stop())
    }
  }, [])

  const stopStream = useCallback(() => {
    streamRef.current?.getTracks().forEach(t => t.stop())
  }, [])

  /* Capture frame → Blob → File */
  const capture = useCallback(() => {
    const video = videoRef.current
    if (!video) return
    const canvas = document.createElement('canvas')
    canvas.width  = video.videoWidth
    canvas.height = video.videoHeight
    canvas.getContext('2d').drawImage(video, 0, 0)
    canvas.toBlob((blob) => {
      if (!blob) return
      const file = new File([blob], `photo_${Date.now()}.jpg`, { type: 'image/jpeg' })
      stopStream()
      onCapture(file)
      onClose()
    }, 'image/jpeg', 0.92)
  }, [onCapture, onClose, stopStream])

  const handleClose = () => { stopStream(); onClose() }

  return (
    /* Overlay */
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
      <div className="relative w-full max-w-2xl bg-forest-900 border border-forest-700 rounded-3xl overflow-hidden shadow-2xl">

        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-forest-800">
          <span className="font-display font-bold text-white">📸 Prendre une photo</span>
          <button onClick={handleClose} className="text-forest-400 hover:text-white transition-colors text-2xl leading-none">&times;</button>
        </div>

        {/* Video */}
        <div className="relative bg-black aspect-video flex items-center justify-center">
          {!ready && !camError && (
            <p className="text-forest-400 text-sm animate-pulse">Initialisation de la caméra…</p>
          )}
          {camError && (
            <p className="text-red-400 text-sm text-center px-6">{camError}</p>
          )}
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className={`w-full h-full object-cover transition-opacity duration-500 ${ready ? 'opacity-100' : 'opacity-0'}`}
          />
          {/* Viewfinder overlay */}
          {ready && (
            <div className="absolute inset-6 border-2 border-fresh/40 rounded-2xl pointer-events-none">
              <span className="absolute top-2 left-2 w-5 h-5 border-t-2 border-l-2 border-fresh rounded-tl-lg" />
              <span className="absolute top-2 right-2 w-5 h-5 border-t-2 border-r-2 border-fresh rounded-tr-lg" />
              <span className="absolute bottom-2 left-2 w-5 h-5 border-b-2 border-l-2 border-fresh rounded-bl-lg" />
              <span className="absolute bottom-2 right-2 w-5 h-5 border-b-2 border-r-2 border-fresh rounded-br-lg" />
            </div>
          )}
        </div>

        {/* Shutter */}
        <div className="flex items-center justify-center gap-4 py-5 px-5">
          <button onClick={handleClose} className="btn-outline px-5 py-2.5">
            Annuler
          </button>
          <button
            onClick={capture}
            disabled={!ready}
            className="btn-primary px-8 py-3 disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:scale-100"
          >
            📷 Capturer
          </button>
        </div>
      </div>
    </div>
  )
}

/* ─── Main Page ──────────────────────────────────────────────────────────── */
export default function AnalyzePage() {
  const [preview, setPreview]       = useState(null)
  const [file, setFile]             = useState(null)
  const [result, setResult]         = useState(null)
  const [loading, setLoading]       = useState(false)
  const [error, setError]           = useState(null)
  const [dragging, setDragging]     = useState(false)
  const [showCamera, setShowCamera] = useState(false)
  const [hasCamera, setHasCamera]   = useState(false)
  const inputRef = useRef(null)

  /* Detect camera availability */
  useEffect(() => {
    if (navigator.mediaDevices?.getUserMedia) setHasCamera(true)
  }, [])

  const reset = () => {
    setPreview(null); setFile(null); setResult(null); setError(null); setLoading(false)
  }

  const handleFile = useCallback((f) => {
    if (!f || !f.type.startsWith('image/')) {
      setError('Veuillez sélectionner un fichier image (JPG, PNG, WEBP).')
      return
    }
    setFile(f)
    setResult(null)
    setError(null)
    setPreview(URL.createObjectURL(f))
  }, [])

  const handleDrop = (e) => {
    e.preventDefault(); setDragging(false)
    handleFile(e.dataTransfer.files[0])
  }

  const handleAnalyze = async () => {
    if (!file) return
    setLoading(true); setError(null); setResult(null)
    try {
      const form = new FormData()
      form.append('file', file)
      const res = await fetch(`${API_BASE}/api/predict`, { method: 'POST', body: form })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Erreur serveur')
      }
      setResult(await res.json())
    } catch (e) {
      setError(`❌ ${e.message}`)
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      {/* Camera Modal */}
      {showCamera && (
        <CameraModal
          onCapture={handleFile}
          onClose={() => setShowCamera(false)}
        />
      )}

      <main className="pt-24 pb-4 min-h-screen px-4 flex flex-col">
        <div className="max-w-2xl mx-auto w-full flex-1">

          {/* Header */}
          <div className="text-center mb-8 animate-fade-up">
            <h1 className="font-display font-black text-4xl md:text-5xl text-white mb-3">
              🔬 Analyser un fruit
            </h1>
            <p className="text-forest-300 text-lg">
              Uploadez une image ou prenez une photo directement depuis votre appareil.
            </p>
          </div>

          {/* Mode selectors (Camera / Upload) */}
          {!file && !loading && !result && (
            <div className="flex gap-3 justify-center mb-6 animate-fade-up">
              {hasCamera && (
                <button
                  onClick={() => setShowCamera(true)}
                  className="btn-primary"
                >
                  📷 Prendre une photo
                </button>
              )}
              <button
                onClick={() => inputRef.current?.click()}
                className="btn-secondary"
              >
                📁 Choisir un fichier
              </button>
              <input
                ref={inputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => handleFile(e.target.files[0])}
              />
            </div>
          )}

          {/* Drop Zone (visible only when no file selected) */}
          {!file && !loading && !result && (
            <div
              className={`drop-zone mb-6 animate-fade-up ${dragging ? 'active' : ''}`}
              onClick={() => inputRef.current?.click()}
              onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
            >
              <div className="flex flex-col items-center gap-4 text-forest-400">
                <span className="text-6xl">🍎</span>
                <div>
                  <p className="text-white font-semibold text-lg">Glissez une image ici</p>
                  <p className="text-sm">ou utilisez les boutons ci-dessus</p>
                </div>
                <p className="text-xs text-forest-600">JPG, PNG, WEBP — max 10 Mo</p>
              </div>
            </div>
          )}

          {/* Image preview (after file selected) */}
          {file && !loading && !result && (
            <div className="mb-6 animate-fade-up card flex flex-col items-center gap-4">
              <img
                src={preview}
                alt="aperçu"
                className="max-h-72 rounded-2xl object-contain border border-forest-700"
              />
              <p className="text-forest-400 text-sm">{file.name}</p>
            </div>
          )}

          {/* Loader */}
          {loading && <ScanLoader />}

          {/* Error */}
          {error && (
            <div className="bg-red-900/30 border border-red-500/40 rounded-2xl p-4 mb-6 text-red-300 text-sm animate-slide-in">
              {error}
            </div>
          )}

          {/* Action buttons */}
          {!loading && (
            <div className="flex flex-wrap gap-3 justify-center mb-8">
              {file && !result && (
                <button onClick={handleAnalyze} className="btn-primary text-base px-8 py-3">
                  ⚡ Analyser maintenant
                </button>
              )}
              {(file || result) && (
                <button onClick={reset} className="btn-outline text-base">
                  🔄 Recommencer
                </button>
              )}
            </div>
          )}

          {/* Result */}
          {result && <ResultCard result={result} preview={preview} />}
        </div>

        <Footer />
      </main>
    </>
  )
}
