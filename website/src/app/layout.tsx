import type { Metadata } from 'next'
import { Navbar } from '@/components'
import './globals.css'

export const metadata: Metadata = {
  title: 'KAMBA MOV - Visão Computacional & IA',
  description: 'Soluções inovadoras em visão computacional e inteligência artificial para transformar seu negócio',
  viewport: 'width=device-width, initial-scale=1',
  keywords: 'IA, visão computacional, machine learning, inteligência artificial, análise de dados',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="pt-BR">
      <body className="bg-primary text-white">
        {children}
      </body>
    </html>
  )
}
