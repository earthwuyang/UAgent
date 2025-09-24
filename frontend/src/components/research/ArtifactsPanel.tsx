import React, { useEffect, useState } from 'react'
import { HTTP_BASE_URL } from '../../config'

type Entry = { name: string; path: string; is_dir: boolean; size: number; modified: number }

const ArtifactsPanel: React.FC<{ sessionId: string; defaultPath?: string }> = ({ sessionId, defaultPath = '' }) => {
  const [path, setPath] = useState<string>(defaultPath)
  const [entries, setEntries] = useState<Entry[]>([])
  const [preview, setPreview] = useState<{ path: string; text: string; truncated: boolean } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const list = async (p: string) => {
    setLoading(true)
    setError(null)
    setPreview(null)
    try {
      const qs = p ? `?path=${encodeURIComponent(p)}` : ''
      const res = await fetch(`${HTTP_BASE_URL}/api/artifacts/${sessionId}/list${qs}`)
      if (!res.ok) throw new Error(`List failed: ${res.status}`)
      const data = await res.json()
      setEntries(data.entries || [])
      setPath(data.path || p)
    } catch (e: any) {
      setError(e?.message || 'Failed to list artifacts')
    } finally {
      setLoading(false)
    }
  }

  const read = async (p: string) => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(`${HTTP_BASE_URL}/api/artifacts/${sessionId}/read?path=${encodeURIComponent(p)}`)
      if (!res.ok) throw new Error(`Read failed: ${res.status}`)
      const data = await res.json()
      setPreview({ path: data.path, text: data.preview || '(binary or empty)', truncated: !!data.truncated })
    } catch (e: any) {
      setError(e?.message || 'Failed to read file')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    list(defaultPath)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId])

  const parentPath = (() => {
    if (!path) return ''
    const parts = path.split('/').filter(Boolean)
    parts.pop()
    return parts.join('/')
  })()

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center gap-2 p-3 border-b bg-gray-50">
        <button className="text-xs px-2 py-1 border rounded bg-white" onClick={() => list('')}>/</button>
        {path && (
          <>
            <span className="text-gray-400">/</span>
            <button className="text-xs px-2 py-1 border rounded bg-white" onClick={() => list(parentPath || '')}>{parentPath || '(root)'}</button>
            <span className="text-gray-400">/</span>
            <span className="text-xs">{path.split('/').filter(Boolean).slice(-1)[0]}</span>
          </>
        )}
        <div className="ml-auto text-xs text-gray-500">{loading ? 'Loading…' : entries.length + ' entries'}</div>
      </div>
      {error && (
        <div className="p-2 text-xs text-red-700 bg-red-50 border-b border-red-200">{error}</div>
      )}
      <div className="flex-1 grid grid-cols-2 gap-0 min-h-0">
        <div className="border-r overflow-auto">
          <table className="w-full text-xs">
            <thead className="sticky top-0 bg-white border-b">
              <tr>
                <th className="text-left px-2 py-1">Name</th>
                <th className="text-right px-2 py-1">Size</th>
              </tr>
            </thead>
            <tbody>
            {entries.map((e) => (
              <tr key={e.path} className="hover:bg-gray-50 cursor-pointer" onClick={() => e.is_dir ? list(e.path) : read(e.path)}>
                <td className="px-2 py-1">
                  {e.is_dir ? '📁' : '📄'} <span className="font-mono">{e.name}</span>
                </td>
                <td className="px-2 py-1 text-right">{e.is_dir ? '-' : e.size}</td>
              </tr>
            ))}
            {entries.length === 0 && (
              <tr><td className="px-2 py-3 text-gray-500" colSpan={2}>No entries</td></tr>
            )}
            </tbody>
          </table>
        </div>
        <div className="overflow-auto">
          {preview ? (
            <div className="p-3">
              <div className="text-xs text-gray-600 mb-2">{preview.path} {preview.truncated && <span className="text-orange-600">(truncated)</span>}</div>
              <pre className="text-[11px] bg-gray-50 p-3 rounded border whitespace-pre-wrap">{preview.text}</pre>
            </div>
          ) : (
            <div className="h-full flex items-center justify-center text-gray-400 text-sm">Select a file to preview</div>
          )}
        </div>
      </div>
    </div>
  )
}

export default ArtifactsPanel

