import { Link, useLocation } from 'react-router-dom'

export default function Navbar() {
  const { pathname } = useLocation()

  const links = [
    { to: '/',            label: 'Accueil' },
    { to: '/analyze',     label: '🔬 Analyser' },
    { to: '/performance', label: '📊 Performances IA' },
  ]

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-forest-950/80 backdrop-blur-md border-b border-forest-800/50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex items-center justify-between h-16">
        {/* Logo */}
        <Link to="/" className="flex items-center gap-2 font-display font-bold text-xl text-white hover:text-fresh transition-colors">
          <span className="text-2xl">🍏</span>
          <span className="hidden sm:block">FreshScan<span className="text-fresh"> AI</span></span>
        </Link>

        {/* Links */}
        <div className="flex items-center gap-1">
          {links.map(({ to, label }) => (
            <Link
              key={to}
              to={to}
              className={`nav-link ${pathname === to ? 'active' : ''}`}
            >
              {label}
            </Link>
          ))}
        </div>

        {/* CTA */}
        <Link to="/analyze" className="btn-primary text-sm py-2 px-4 hidden md:inline-flex">
          Analyser gratuitement
        </Link>
      </div>
    </nav>
  )
}
