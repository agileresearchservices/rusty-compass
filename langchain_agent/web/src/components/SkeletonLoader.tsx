/**
 * SkeletonLoader - Shows a loading placeholder while content is being fetched
 */

import clsx from 'clsx'

interface SkeletonLoaderProps {
  count?: number
  height?: string
  width?: string
  className?: string
}

export function SkeletonLoader({
  count = 1,
  height = 'h-4',
  width = 'w-full',
  className = '',
}: SkeletonLoaderProps) {
  return (
    <>
      {Array.from({ length: count }).map((_, i) => (
        <div
          key={i}
          className={clsx(
            'bg-gray-700 rounded animate-pulse',
            height,
            width,
            className
          )}
          aria-hidden="true"
        />
      ))}
    </>
  )
}

interface SkeletonConversationItemProps {
  count?: number
}

export function SkeletonConversationItem({ count = 3 }: SkeletonConversationItemProps) {
  return (
    <div className="space-y-1 px-2">
      {Array.from({ length: count }).map((_, i) => (
        <div
          key={i}
          className="w-full flex items-start gap-2 px-3 py-2 rounded-lg"
          aria-hidden="true"
        >
          <div className="w-4 h-4 mt-0.5 flex-shrink-0 bg-gray-700 rounded animate-pulse" />
          <div className="flex-1 min-w-0 space-y-2">
            <div className="h-4 bg-gray-700 rounded animate-pulse w-3/4" />
            <div className="h-3 bg-gray-700 rounded animate-pulse w-1/2" />
          </div>
        </div>
      ))}
    </div>
  )
}
