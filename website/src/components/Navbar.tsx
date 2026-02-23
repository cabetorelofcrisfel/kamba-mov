'use client'

import Link from 'next/link'
import { useState } from 'react'

export default function Navbar() {
  const [isOpen, setIsOpen] = useState(false)

  return (
    <nav className="fixed w-full bg-primary/80 backdrop-blur-md z-50 border-b border-secondary/20">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center">
            <span className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-secondary to-accent">
              KAMBA
            </span>
            <span className="text-2xl font-bold text-white ml-1">MOV</span>
          </div>

          <div className="hidden md:flex items-center space-x-8">
            <Link href="#home" className="text-gray-300 hover:text-secondary transition">
              Home
            </Link>
            <Link href="#features" className="text-gray-300 hover:text-secondary transition">
              Recursos
            </Link>
            <Link href="#about" className="text-gray-300 hover:text-secondary transition">
              Sobre
            </Link>
            <Link href="#contact" className="text-gray-300 hover:text-secondary transition">
              Contato
            </Link>
          </div>

          <button
            onClick={() => setIsOpen(!isOpen)}
            className="md:hidden text-secondary p-2"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
        </div>

        {isOpen && (
          <div className="md:hidden pb-4 space-y-2">
            <Link href="#home" className="block text-gray-300 hover:text-secondary py-2">
              Home
            </Link>
            <Link href="#features" className="block text-gray-300 hover:text-secondary py-2">
              Recursos
            </Link>
            <Link href="#about" className="block text-gray-300 hover:text-secondary py-2">
              Sobre
            </Link>
            <Link href="#contact" className="block text-gray-300 hover:text-secondary py-2">
              Contato
            </Link>
          </div>
        )}
      </div>
    </nav>
  )
}
