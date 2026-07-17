import { useState } from 'react'
import Footer from '../components/Footer'

const API_BASE = 'http://localhost:8000'

/* ─── Static data (descriptions) ─────────────────────────────────────────── */
const PHASES = [
  {
    id: 'scratch',
    label: 'Phase 1 — CNN from scratch',
    tag: 'Baseline',
    tagColor: 'bg-red-500/20 text-red-400 border-red-500/30',
    description:
      'Première architecture personnalisée avec 3 blocs Conv2D + MaxPooling. ' +
      'La couche Dense finale contient à elle seule 4.2M de paramètres sur 4.3M au total, ' +
      'causant un surapprentissage (overfitting) massif. La validation accuracy plafonne à 87%.',
    plots: ['curves_cnn_scratch_multi-classes.png'],
    metric: { label: 'Val. Accuracy', value: '87%', color: 'text-red-400' },
  },
  {
    id: 'augmented',
    label: 'Phase 2 — Régularisation',
    tag: 'Data Aug. + Dropout',
    tagColor: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
    description:
      'Ajout de transformations aléatoires (zoom, rotation, flip) et d\'une couche Dropout (40%). ' +
      'L\'écart train/val se réduit nettement, mais le réseau trop peu profond (3 blocs seulement) ' +
      'ne parvient pas à dépasser ~91% de validation accuracy sur 28 classes.',
    plots: ['curves_cnn_augmenté_+_dropout.png'],
    metric: { label: 'Val. Accuracy', value: '91%', color: 'text-yellow-400' },
  },
  {
    id: 'tl_head',
    label: 'Phase 3a — Transfer Learning (tête seule)',
    tag: 'MobileNetV2 gelé',
    tagColor: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
    description:
      'Base MobileNetV2 pré-entraînée sur ImageNet (2.2M paramètres gelés). ' +
      'On entraîne uniquement la nouvelle tête de classification (Dense 128 → Softmax 28). ' +
      'Dès la première epoch, la val_accuracy dépasse 94% — la preuve de la puissance du Transfer Learning.',
    plots: ['curves_transfer_learning_-_tête_seule.png'],
    metric: { label: 'Val. Accuracy', value: '97%', color: 'text-blue-400' },
  },
  {
    id: 'finetuning',
    label: 'Phase 3b — Fine-Tuning',
    tag: '🏆 Meilleur modèle',
    tagColor: 'bg-fresh/20 text-fresh border-fresh/30',
    description:
      'Dégel des 20% dernières couches de MobileNetV2 avec un learning rate très faible (1e-4). ' +
      'Le réseau affine ses représentations visuelles pour nos 28 classes spécifiques. ' +
      'Résultat : 98% de val_accuracy. Le modèle est ensuite converti en TFLite (9 Mo FP32, 2.7 Mo INT8).',
    plots: ['curves_transfer_learning_-_fine-tuning.png'],
    metric: { label: 'Val. Accuracy', value: '98%', color: 'text-fresh' },
  },
]

const KPI_CARDS = [
  { icon: '🎯', label: 'Précision Finale', value: '98%', sub: 'sur 5 863 images de validation' },
  { icon: '⚡', label: 'Taille du Modèle', value: '2.7 Mo', sub: 'TFLite INT8 quantifié' },
  { icon: '🧠', label: 'Paramètres (base)', value: '2.2M', sub: 'MobileNetV2 pré-entraîné' },
  { icon: '🏷️', label: 'Classes détectées', value: '28', sub: '14 fruits × 2 états' },
]

/* ─── Image viewer ─────────────────────────────────────────────────────────── */
function PlotImage({ filename }) {
  const [loaded, setLoaded] = useState(false)
  const url = `${API_BASE}/plots/${filename}`

  return (
    <div className="relative overflow-hidden rounded-2xl bg-forest-800/50 border border-forest-700/40">
      {!loaded && (
        <div className="absolute inset-0 flex items-center justify-center text-forest-500 text-sm animate-pulse">
          Chargement du graphique…
        </div>
      )}
      <img
        src={url}
        alt={filename}
        onLoad={() => setLoaded(true)}
        className={`w-full object-contain transition-opacity duration-500 ${loaded ? 'opacity-100' : 'opacity-0'}`}
      />
    </div>
  )
}

/* ─── Phase tab ─────────────────────────────────────────────────────────────── */
function PhaseTab({ phase, selected, onClick }) {
  return (
    <button
      onClick={onClick}
      className={`flex-1 min-w-0 px-3 py-2.5 rounded-xl text-xs md:text-sm font-medium transition-all duration-200 text-left
        ${selected
          ? 'bg-forest-700 text-white shadow-md'
          : 'text-forest-400 hover:text-white hover:bg-forest-800/60'}`}
    >
      <span className="block truncate">{phase.label}</span>
    </button>
  )
}

/* ─── Main Page ─────────────────────────────────────────────────────────────── */
export default function PerformancePage() {
  const [activePhase, setActivePhase] = useState(0)
  const phase = PHASES[activePhase]

  return (
    <main className="pt-24 pb-20 min-h-screen px-4">
      <div className="max-w-5xl mx-auto">

        {/* Header */}
        <div className="text-center mb-14 animate-fade-up">
          <div className="inline-flex items-center gap-2 bg-fresh/10 border border-fresh/30 rounded-full px-4 py-1.5 text-fresh text-sm font-medium mb-6">
            📊 Page Portfolio &amp; R&amp;D
          </div>
          <h1 className="font-display font-black text-4xl md:text-5xl text-white mb-4">
            Performances du Modèle IA
          </h1>
          <p className="text-forest-300 text-lg max-w-2xl mx-auto">
            Évolution pas à pas de l'entraînement du CNN, du baseline naïf jusqu'au Transfer Learning
            MobileNetV2 fine-tuné à <strong className="text-fresh">98% de précision</strong>.
          </p>
        </div>

        {/* KPI Cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-14">
          {KPI_CARDS.map(({ icon, label, value, sub }) => (
            <div key={label} className="kpi-card">
              <div className="text-3xl mb-2">{icon}</div>
              <div className="font-display font-black text-2xl md:text-3xl text-fresh mb-1">{value}</div>
              <p className="font-semibold text-white text-xs md:text-sm">{label}</p>
              <p className="text-forest-400 text-xs mt-1">{sub}</p>
            </div>
          ))}
        </div>

        {/* Phases section */}
        <div className="card mb-10">
          <h2 className="font-display font-bold text-2xl text-white mb-6">📈 Évolution de l'Entraînement</h2>

          {/* Tabs */}
          <div className="flex flex-col sm:flex-row gap-2 mb-8 bg-forest-900/60 rounded-2xl p-2">
            {PHASES.map((p, i) => (
              <PhaseTab key={p.id} phase={p} selected={activePhase === i} onClick={() => setActivePhase(i)} />
            ))}
          </div>

          {/* Active phase content */}
          <div className="animate-fade-up" key={phase.id}>
            <div className="flex flex-wrap items-center gap-3 mb-4">
              <h3 className="font-semibold text-white text-lg">{phase.label}</h3>
              <span className={`border rounded-full px-3 py-0.5 text-xs font-semibold ${phase.tagColor}`}>
                {phase.tag}
              </span>
              <span className={`font-display font-black text-2xl ${phase.metric.color}`}>
                {phase.metric.value}
              </span>
              <span className="text-forest-500 text-xs">{phase.metric.label}</span>
            </div>
            <p className="text-forest-300 text-sm leading-relaxed mb-6">{phase.description}</p>

            <div className="space-y-4">
              {phase.plots.map(f => <PlotImage key={f} filename={f} />)}
            </div>
          </div>
        </div>

        {/* Comparison & Confusion Matrix */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-10">
          <div className="card">
            <h3 className="font-display font-bold text-xl text-white mb-4">📊 Comparaison Globale</h3>
            <p className="text-forest-300 text-sm mb-4">
              Graphique récapitulatif des performances des 3 approches sur les mêmes données.
            </p>
            <PlotImage filename="comparison_all.png" />
          </div>

          <div className="card">
            <h3 className="font-display font-bold text-xl text-white mb-4">🔵 Matrice de Confusion</h3>
            <p className="text-forest-300 text-sm mb-4">
              Performance détaillée sur les 28 classes — le modèle distingue parfaitement
              fruits frais et avariés, même entre variétés visuellement proches.
            </p>
            <PlotImage filename="confusion_matrix.png" />
          </div>
        </div>

        {/* Tech stack */}
        <div className="card text-center">
          <h3 className="font-display font-bold text-xl text-white mb-6">⚙️ Stack Technique</h3>
          <div className="flex flex-wrap justify-center gap-3">
            {[
              'Python 3.12', 'TensorFlow 2.x', 'MobileNetV2', 'Transfer Learning',
              'TFLite FP32 / INT8', 'FastAPI', 'React', 'Tailwind CSS', 'Vite',
            ].map(t => (
              <span key={t} className="bg-forest-800 border border-forest-700 text-forest-200 text-xs px-3 py-1.5 rounded-full">
                {t}
              </span>
            ))}
          </div>
        </div>

        <Footer />
      </div>
    </main>
  )
}
