import { useState, useEffect, useCallback } from 'react'
import ImageUploader from './components/ImageUploader/ImageUploader'
import ResultPanel from './components/ResultPanel/ResultPanel'
import DetectionPanel from './components/DetectionPanel/DetectionPanel'
import StatusBar from './components/StatusBar/StatusBar'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

export default function App() {
  const [health, setHealth] = useState(null)
  const [imageFile, setImageFile] = useState(null)
  const [imagePreviewUrl, setImagePreviewUrl] = useState(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [classifyResult, setClassifyResult] = useState(null)
  const [detectResult, setDetectResult] = useState(null)
  const [classifyError, setClassifyError] = useState(null)
  const [detectError, setDetectError] = useState(null)
  const [detectUnavailable, setDetectUnavailable] = useState(false)

  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const res = await fetch(`${API_BASE}/health`)
        if (res.ok) {
          const data = await res.json()
          setHealth(data)
        }
      } catch {
        setHealth(null)
      }
    }
    fetchHealth()
    const interval = setInterval(fetchHealth, 15000)
    return () => clearInterval(interval)
  }, [])

  const handleImageSelect = useCallback((file, previewUrl) => {
    setImageFile(file)
    setImagePreviewUrl(previewUrl)
    setClassifyResult(null)
    setDetectResult(null)
    setClassifyError(null)
    setDetectError(null)
    setDetectUnavailable(false)
  }, [])

  const handleAnalyze = async () => {
    if (!imageFile) return

    setIsAnalyzing(true)
    setClassifyResult(null)
    setDetectResult(null)
    setClassifyError(null)
    setDetectError(null)
    setDetectUnavailable(false)

    const formDataClassify = new FormData()
    formDataClassify.append('file', imageFile)

    const formDataDetect = new FormData()
    formDataDetect.append('file', imageFile)

    const [classifyRes, detectRes] = await Promise.allSettled([
      fetch(`${API_BASE}/predict`, { method: 'POST', body: formDataClassify }),
      fetch(`${API_BASE}/detect`, { method: 'POST', body: formDataDetect }),
    ])

    if (classifyRes.status === 'fulfilled') {
      const res = classifyRes.value
      if (res.ok) {
        setClassifyResult(await res.json())
      } else {
        setClassifyError(`Classification failed (${res.status})`)
      }
    } else {
      setClassifyError('Could not reach the classification endpoint.')
    }

    if (detectRes.status === 'fulfilled') {
      const res = detectRes.value
      if (res.status === 503) {
        setDetectUnavailable(true)
      } else if (res.ok) {
        setDetectResult(await res.json())
      } else {
        setDetectError(`Detection failed (${res.status})`)
      }
    } else {
      setDetectError('Could not reach the detection endpoint.')
    }

    setIsAnalyzing(false)
  }

  const hasResults = classifyResult || detectResult || classifyError || detectError || detectUnavailable

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="app-header-logo">
          <div className="logo-mark">
            <svg viewBox="0 0 14 14" aria-hidden="true">
              <rect x="1" y="1" width="5" height="5" rx="1" />
              <rect x="8" y="1" width="5" height="5" rx="1" />
              <rect x="1" y="8" width="5" height="5" rx="1" />
              <rect x="8" y="8" width="5" height="5" rx="1" />
            </svg>
          </div>
          <div>
            <div className="logo-title">DefectNet</div>
            <div className="logo-subtitle">Surface Defect Detection System</div>
          </div>
        </div>
        <StatusBar health={health} />
      </header>

      <main className="app-main">
        <div className="app-layout-top">
          <ImageUploader
            imageFile={imageFile}
            imagePreviewUrl={imagePreviewUrl}
            onImageSelect={handleImageSelect}
            onAnalyze={handleAnalyze}
            isAnalyzing={isAnalyzing}
          />
          <DetectionPanel
            result={detectResult}
            error={detectError}
            unavailable={detectUnavailable}
            originalImageUrl={imagePreviewUrl}
            isAnalyzing={isAnalyzing}
          />
        </div>

        {(hasResults || isAnalyzing) && (
          <div className="app-layout-bottom">
            <ResultPanel
              result={classifyResult}
              error={classifyError}
              isAnalyzing={isAnalyzing}
            />
          </div>
        )}
      </main>
    </div>
  )
}
