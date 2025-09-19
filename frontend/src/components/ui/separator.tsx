import React from 'react'

interface SeparatorProps {
  className?: string
}

export const Separator: React.FC<SeparatorProps> = ({ className = '' }) => {
  return (
    <div className={`border-b border-gray-200 ${className}`} />
  )
}