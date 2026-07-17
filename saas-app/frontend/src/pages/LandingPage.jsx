import { Link } from 'react-router-dom'
import Footer from '../components/Footer'

const stats = [
  { value: '1/3', label: 'des aliments produits sont gaspillés chaque année dans le monde' },
  { value: '1.3 Md t', label: 'de nourriture jetée annuellement selon la FAO' },
  { value: '8-10%', label: 'des émissions de gaz à effet de serre liées au gaspillage' },
]

const features = [
  {
    icon: '⚡',
    title: 'Détection Instantanée',
    desc: 'Notre CNN MobileNetV2 analyse votre fruit en moins d\'une seconde avec 98% de précision.',
  },
  {
    icon: '🍽️',
    title: 'Recettes Anti-Gaspi',
    desc: 'Pour chaque fruit abîmé, nous proposons des recettes simples pour éviter le gaspillage.',
  },
  {
    icon: '💡',
    title: 'Astuces Conservation',
    desc: 'Pour les fruits frais, obtenez les meilleures techniques pour maximiser leur durée de vie.',
  },
  {
    icon: '📱',
    title: '100% Mobile',
    desc: 'Prenez une photo depuis votre téléphone et obtenez le résultat en temps réel.',
  },
]

const fruits = ['🍎', '🍌', '🫑', '🥕', '🥒', '🍇', '🥭', '🍊', '🍓', '🍅', '🥔', '🍎']

export default function LandingPage() {
  return (
    <main className="pt-16 overflow-hidden">

      {/* ── Hero ─────────────────────────────────────────────────────────── */}
      <section className="relative min-h-[92vh] flex items-center justify-center text-center px-4">
        <div className="hero-glow absolute inset-0 pointer-events-none" />

        {/* Floating fruit emojis */}
        <div className="absolute inset-0 pointer-events-none select-none overflow-hidden">
          {fruits.map((f, i) => (
            <span
              key={i}
              className="absolute text-4xl opacity-10 animate-float"
              style={{
                left: `${(i * 9) % 95}%`,
                top:  `${(i * 13 + 5) % 85}%`,
                animationDelay: `${i * 0.3}s`,
                animationDuration: `${3 + (i % 3)}s`,
              }}
            >
              {f}
            </span>
          ))}
        </div>

        <div className="relative z-10 max-w-4xl mx-auto animate-fade-up">
          <div className="inline-flex items-center gap-2 bg-fresh/10 border border-fresh/30 rounded-full px-4 py-1.5 text-fresh text-sm font-medium mb-8">
            <span className="w-2 h-2 rounded-full bg-fresh animate-pulse inline-block" />
            SaaS gratuit • Intelligence Artificielle • 28 classes
          </div>

          <h1 className="font-display font-black text-5xl md:text-7xl lg:text-8xl leading-none mb-6">
            Ne jetez plus<br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-fresh-light via-fresh to-forest-400">
              vos fruits abîmés
            </span>
          </h1>

          <p className="text-forest-300 text-xl md:text-2xl max-w-2xl mx-auto mb-10 leading-relaxed">
            Photographiez un fruit ou légume. Notre IA détecte instantanément s'il est frais ou avarié
            et vous suggère <strong className="text-white">recettes</strong> ou <strong className="text-white">astuces de conservation</strong>.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link to="/analyze" className="btn-primary text-lg px-8 py-4">
              🔬 Analyser un fruit gratuitement
            </Link>
            <Link to="/performance" className="btn-secondary text-lg px-8 py-4">
              📊 Voir les performances du modèle
            </Link>
          </div>

          <p className="text-forest-500 text-sm mt-6">
            Modèle MobileNetV2 • 98% de précision • 28 classes de fruits & légumes
          </p>
        </div>
      </section>

      {/* ── Stats ──────────────────────────────────────────────────────────── */}
      <section className="py-20 px-4 bg-forest-900/30">
        <div className="max-w-5xl mx-auto">
          <p className="text-center text-forest-400 text-sm uppercase tracking-widest mb-12 font-semibold">
            Le gaspillage alimentaire en chiffres
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {stats.map(({ value, label }) => (
              <div key={value} className="kpi-card">
                <div className="font-display font-black text-4xl text-fresh mb-3">{value}</div>
                <p className="text-forest-300 text-sm leading-relaxed">{label}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Features ───────────────────────────────────────────────────────── */}
      <section className="py-24 px-4">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="section-title mb-4">Tout ce dont vous avez besoin</h2>
            <p className="section-subtitle max-w-xl mx-auto">
              Un outil simple, gratuit et puissant pour lutter contre le gaspillage alimentaire au quotidien.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {features.map(({ icon, title, desc }) => (
              <div
                key={title}
                className="card group hover:border-fresh/40 hover:-translate-y-2 transition-all duration-300"
              >
                <div className="text-4xl mb-4 group-hover:scale-110 transition-transform duration-300">{icon}</div>
                <h3 className="font-display font-bold text-lg text-white mb-2">{title}</h3>
                <p className="text-forest-300 text-sm leading-relaxed">{desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── CTA Banner ─────────────────────────────────────────────────────── */}
      <section className="py-24 px-4 bg-gradient-to-r from-forest-900 via-forest-800 to-forest-900">
        <div className="max-w-3xl mx-auto text-center">
          <h2 className="section-title mb-4">Prêt à zéro gaspi ?</h2>
          <p className="section-subtitle mb-10">
            Commencez maintenant gratuitement. Aucune inscription requise.
          </p>
          <Link to="/analyze" className="btn-primary text-xl px-10 py-5">
            🚀 Lancer l'analyse IA
          </Link>
        </div>
      </section>

      <Footer />
    </main>
  )
}
