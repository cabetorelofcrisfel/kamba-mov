'use client'

import { motion } from 'framer-motion'
import { useState } from 'react'

export default function Contact() {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    message: '',
  })

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: value,
    }))
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    console.log('Form submitted:', formData)
    alert('Obrigado pelo seu contato! Em breve retornaremos.')
    setFormData({ name: '', email: '', message: '' })
  }

  return (
    <section id="contact" className="py-20 bg-primary relative overflow-hidden">
      <div className="absolute inset-0">
        <div className="absolute w-96 h-96 bg-secondary/10 rounded-full blur-3xl -bottom-20 -right-20 animate-pulse"></div>
      </div>

      <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="text-center mb-12"
        >
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
            Entre em Contato
          </h2>
          <p className="text-gray-400 text-lg">
            Vamos conversar sobre como podemos ajudar você
          </p>
        </motion.div>

        <motion.form
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          viewport={{ once: true }}
          onSubmit={handleSubmit}
          className="space-y-6 bg-gradient-to-br from-gray-900 to-gray-800 p-8 rounded-xl border border-secondary/20"
        >
          <div>
            <label htmlFor="name" className="block text-white font-semibold mb-2">
              Nome
            </label>
            <input
              type="text"
              id="name"
              name="name"
              value={formData.name}
              onChange={handleChange}
              required
              className="w-full px-4 py-3 rounded-lg bg-gray-700 text-white border border-gray-600 focus:border-secondary focus:outline-none transition"
              placeholder="Seu nome completo"
            />
          </div>

          <div>
            <label htmlFor="email" className="block text-white font-semibold mb-2">
              Email
            </label>
            <input
              type="email"
              id="email"
              name="email"
              value={formData.email}
              onChange={handleChange}
              required
              className="w-full px-4 py-3 rounded-lg bg-gray-700 text-white border border-gray-600 focus:border-secondary focus:outline-none transition"
              placeholder="seu.email@exemplo.com"
            />
          </div>

          <div>
            <label htmlFor="message" className="block text-white font-semibold mb-2">
              Mensagem
            </label>
            <textarea
              id="message"
              name="message"
              value={formData.message}
              onChange={handleChange}
              required
              rows={5}
              className="w-full px-4 py-3 rounded-lg bg-gray-700 text-white border border-gray-600 focus:border-secondary focus:outline-none transition resize-none"
              placeholder="Sua mensagem..."
            />
          </div>

          <button
            type="submit"
            className="w-full py-3 bg-gradient-to-r from-secondary to-accent text-white font-semibold rounded-lg hover:shadow-lg transition transform hover:scale-105"
          >
            Enviar Mensagem
          </button>
        </motion.form>

        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          viewport={{ once: true }}
          className="mt-12 grid md:grid-cols-3 gap-6"
        >
          <div className="text-center">
            <div className="text-3xl mb-2">📧</div>
            <p className="text-gray-300">contato@kambamov.com</p>
          </div>
          <div className="text-center">
            <div className="text-3xl mb-2">📍</div>
            <p className="text-gray-300">São Paulo, Brasil</p>
          </div>
          <div className="text-center">
            <div className="text-3xl mb-2">📞</div>
            <p className="text-gray-300">(11) 9999-9999</p>
          </div>
        </motion.div>
      </div>
    </section>
  )
}
