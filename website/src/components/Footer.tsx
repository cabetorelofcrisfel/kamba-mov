export default function Footer() {
  const currentYear = new Date().getFullYear()

  return (
    <footer className="bg-primary border-t border-secondary/20 py-12">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid md:grid-cols-4 gap-8 mb-8">
          <div>
            <h3 className="text-xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-secondary to-accent mb-4">
              KAMBA MOV
            </h3>
            <p className="text-gray-400">
              Soluções em visão computacional e IA
            </p>
          </div>

          <div>
            <h4 className="text-white font-semibold mb-4">Produto</h4>
            <ul className="space-y-2 text-gray-400">
              <li><a href="#" className="hover:text-secondary transition">Recursos</a></li>
              <li><a href="#" className="hover:text-secondary transition">Preços</a></li>
              <li><a href="#" className="hover:text-secondary transition">Documentação</a></li>
            </ul>
          </div>

          <div>
            <h4 className="text-white font-semibold mb-4">Empresa</h4>
            <ul className="space-y-2 text-gray-400">
              <li><a href="#about" className="hover:text-secondary transition">Sobre</a></li>
              <li><a href="#" className="hover:text-secondary transition">Blog</a></li>
              <li><a href="#contact" className="hover:text-secondary transition">Contato</a></li>
            </ul>
          </div>

          <div>
            <h4 className="text-white font-semibold mb-4">Legal</h4>
            <ul className="space-y-2 text-gray-400">
              <li><a href="#" className="hover:text-secondary transition">Privacidade</a></li>
              <li><a href="#" className="hover:text-secondary transition">Termos</a></li>
              <li><a href="#" className="hover:text-secondary transition">LGPD</a></li>
            </ul>
          </div>
        </div>

        <div className="border-t border-gray-700 pt-8">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <p className="text-gray-400 text-sm">
              © {currentYear} KAMBA MOV. Todos os direitos reservados.
            </p>
            <div className="flex gap-6 mt-4 md:mt-0">
              <a href="#" className="text-gray-400 hover:text-secondary transition">
                Twitter
              </a>
              <a href="#" className="text-gray-400 hover:text-secondary transition">
                LinkedIn
              </a>
              <a href="#" className="text-gray-400 hover:text-secondary transition">
                GitHub
              </a>
            </div>
          </div>
        </div>
      </div>
    </footer>
  )
}
