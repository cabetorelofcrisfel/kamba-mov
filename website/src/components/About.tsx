'use client'

import { motion } from 'framer-motion'

export default function About() {
  return (
    <section id="about" className="py-20 bg-gradient-to-b from-primary to-gray-900">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid md:grid-cols-2 gap-12 items-center">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            whileInView={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
          >
            <h2 className="text-4xl font-bold text-white mb-6">
              Sobre a <span className="text-transparent bg-clip-text bg-gradient-to-r from-secondary to-accent">KAMBA MOV</span>
            </h2>
            <p className="text-gray-300 mb-4 text-lg">
              A Kamba Mov é uma empresa especializada em desenvolvimento de soluções inovadoras utilizando visão computacional e inteligência artificial.
            </p>
            <p className="text-gray-300 mb-4 text-lg">
              Com uma equipe de especialistas altamente qualificados, oferecemos serviços de ponta que transformam dados visuais em insights valiosos para seu negócio.
            </p>
            <p className="text-gray-300 text-lg">
              Nossa missão é democratizar o acesso a tecnologias de IA e visão computacional, tornando-as acessíveis e aplicáveis a diversos setores.
            </p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 20 }}
            whileInView={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="relative"
          >
            <div className="bg-gradient-to-br from-secondary/20 to-accent/20 rounded-2xl p-12 border border-secondary/30">
              <div className="space-y-8">
                <div>
                  <h3 className="text-secondary font-bold text-3xl mb-2">500+</h3>
                  <p className="text-gray-300">Projetos Completados</p>
                </div>
                <div>
                  <h3 className="text-accent font-bold text-3xl mb-2">100+</h3>
                  <p className="text-gray-300">Clientes Satisfeitos</p>
                </div>
                <div>
                  <h3 className="text-secondary font-bold text-3xl mb-2">50+</h3>
                  <p className="text-gray-300">Profissionais Dedicados</p>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  )
}
