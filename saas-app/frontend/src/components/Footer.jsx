export default function Footer() {
  const year = new Date().getFullYear()
  return (
    <footer className="py-6 px-4 border-t border-forest-800/50 text-center text-forest-500 text-sm">
      <p>
        🍏{' '}
        <span className="text-forest-400 font-medium">FreshScan AI</span>
        {' — '}Projet Deep Learning · MobileNetV2 · Portfolio
      </p>
      <p className="mt-1 text-forest-600 text-xs">
        © {year}{' '}
        <a
          href="https://github.com/Anas7823"
          target="_blank"
          rel="noopener noreferrer"
          className="hover:text-fresh transition-colors duration-200"
        >
          EL KHIAT Anas
        </a>
        {' '}— Tous droits réservés
      </p>
    </footer>
  )
}
