import React from 'react'

interface AlertProps {
  variant?: 'default' | 'destructive'
  className?: string
  children: React.ReactNode
}

interface AlertDescriptionProps {
  className?: string
  children: React.ReactNode
}

export const Alert: React.FC<AlertProps> = ({
  variant = 'default',
  className = '',
  children
}) => {
  const baseClasses = 'relative w-full rounded-lg border p-4'

  const variantClasses = {
    default: 'bg-white border-gray-200 text-gray-950',
    destructive: 'border-red-200 bg-red-50 text-red-800'
  }

  return (
    <div className={`${baseClasses} ${variantClasses[variant]} ${className}`}>
      {children}
    </div>
  )
}

export const AlertDescription: React.FC<AlertDescriptionProps> = ({ className = '', children }) => {
  return (
    <div className={`text-sm ${className}`}>
      {children}
    </div>
  )
}