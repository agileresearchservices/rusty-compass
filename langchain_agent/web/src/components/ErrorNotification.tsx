/**
 * ErrorNotification - Displays error messages to users
 */

import { AlertCircle, X } from 'lucide-react'
import { useEffect } from 'react'

interface ErrorNotificationProps {
  message: string
  onDismiss: () => void
  autoClose?: boolean
  duration?: number
}

export function ErrorNotification({
  message,
  onDismiss,
  autoClose = true,
  duration = 5000,
}: ErrorNotificationProps) {
  useEffect(() => {
    if (!autoClose) return

    const timer = setTimeout(onDismiss, duration)
    return () => clearTimeout(timer)
  }, [autoClose, duration, onDismiss])

  return (
    <div
      className="flex items-start gap-3 p-4 bg-red-900/20 border border-red-500/30 rounded-lg text-red-200"
      role="alert"
      aria-live="assertive"
    >
      <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" aria-hidden="true" />
      <div className="flex-1">
        <p className="text-sm font-medium">{message}</p>
      </div>
      <button
        onClick={onDismiss}
        aria-label="Dismiss error message"
        className="flex-shrink-0 p-1 text-red-300 hover:text-red-200 focus:outline-none focus:ring-2 focus:ring-red-500 rounded"
      >
        <X className="w-4 h-4" />
      </button>
    </div>
  )
}
