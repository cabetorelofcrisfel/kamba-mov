'use client'

import { motion } from 'framer-motion'

const features = [
  {
    icon: '🤖',
    title: 'IA Avançada',
    description: 'Algoritmos de machine learning de última geração para análise de dados',
  },
  {
    icon: '📹',
    title: 'Visão Computacional',
    description: 'Processamento de imagem e vídeo em tempo real com precisão',
  },
  {
    icon: '⚡',
    title: 'Performance',
    description: 'Soluções otimizadas para máxima velocidade e eficiência',
  },
  {
    icon: '🔒',
    title: 'Segurança',
    description: 'Proteção de dados com criptografia e conformidade LGPD',
  },
  {
    icon: '📊',
    title: 'Analytics',
    description: 'Dashboards intuitivos e relatórios detalhados em tempo real',
  },
  {
    icon: '🌐',
    title: 'Integração',
    description: 'APIs robustas para fácil integração em seus sistemas',
  },
]

export default function Features() {
  return (
    <section id="features" className="py-20 bg-primary relative">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
            Nossos Recursos
          </h2>
          <p className="text-gray-400 text-lg">
            Tecnologia de ponta para impulsionar seu negócio
          </p>
        </motion.div>

        <div className="grid md:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              viewport={{ once: true }}
              className="p-8 rounded-xl bg-gradient-to-br from-gray-900 to-gray-800 border border-secondary/20 hover:border-secondary/50 transition group cursor-pointer"
            >
              <div className="text-5xl mb-4 group-hover:scale-110 transition">{feature.icon}</div>
              <h3 className="text-xl font-bold text-white mb-3">{feature.title}</h3>
              <p className="text-gray-400">{feature.description}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  )
}
