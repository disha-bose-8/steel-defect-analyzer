import { useRef, useState, useCallback } from 'react'
import './ImageUploader.css'

export default function ImageUploader({
  imageFile,
  imagePreviewUrl,
  onImageSelect,
  onAnalyze,
  isAnalyzing,
}) {
  const inputRef = useRef(null)
  const [isDragging, setIsDragging] = useState(false)

  const handleFile = useCallback((file) => {
    if (!file || !file.type.startsWith('image/')) return
    const url = URL.createObjectURL(file)
    onImageSelect(file, url)
  }, [onImageSelect])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files[0]
    handleFile(file)
  }, [handleFile])

  const handleDragOver = (e) => { e.preventDefault(); setIsDragging(true) }
  const handleDragLeave = () => setIsDragging(false)

  const handleInputChange = (e) => {
    handleFile(e.target.files[0])
  }

  const handleClear = (e) => {
    e.stopPropagation()
    onImageSelect(null, null)
    if (inputRef.current) inputRef.current.value = ''
  }

  return (
    <div className="panel uploader-panel">
      <div className="panel-header">
        <svg className="panel-header-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
          <polyline points="17 8 12 3 7 8" />
          <line x1="12" y1="3" x2="12" y2="15" />
        </svg>
        <span className="panel-title">Input Image</span>
      </div>

      <div className="panel-body uploader-body">
        <div
          className={`drop-zone ${isDragging ? 'drop-zone--dragging' : ''} ${imagePreviewUrl ? 'drop-zone--has-image' : ''}`}
          onClick={() => !imagePreviewUrl && inputRef.current?.click()}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          role="button"
          tabIndex={imagePreviewUrl ? -1 : 0}
          aria-label="Upload steel surface image"
          onKeyDown={(e) => e.key === 'Enter' && !imagePreviewUrl && inputRef.current?.click()}
        >
          {imagePreviewUrl ? (
            <div className="preview-container">
              <img
                src={imagePreviewUrl}
                alt="Uploaded steel surface"
                className="preview-image"
              />
              <button
                className="clear-btn"
                onClick={handleClear}
                aria-label="Remove image"
              >
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              </button>
              <div className="preview-meta">
                <span className="preview-filename">{imageFile?.name}</span>
                <button
                  className="change-btn"
                  onClick={(e) => { e.stopPropagation(); inputRef.current?.click() }}
                >
                  Change
                </button>
              </div>
            </div>
          ) : (
            <div className="drop-prompt">
              <div className="drop-icon-wrap">
                <svg viewBox="0 0 40 40" fill="none" stroke="currentColor" strokeWidth="1.2" aria-hidden="true">
                  <rect x="4" y="8" width="32" height="24" rx="3" />
                  <circle cx="13" cy="16" r="3" />
                  <path d="M4 26l8-7 6 5 5-4 9 7" />
                </svg>
              </div>
              <p className="drop-main">Drop a surface image here</p>
              <p className="drop-sub">or click to browse — JPG, PNG, BMP</p>
            </div>
          )}
        </div>

        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="file-input-hidden"
          onChange={handleInputChange}
          aria-hidden="true"
          tabIndex={-1}
        />

        <button
          className={`analyze-btn ${isAnalyzing ? 'analyze-btn--loading' : ''}`}
          onClick={onAnalyze}
          disabled={!imageFile || isAnalyzing}
        >
          {isAnalyzing ? (
            <>
              <span className="spinner" aria-hidden="true" />
              Analyzing…
            </>
          ) : (
            <>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
                <circle cx="11" cy="11" r="8" />
                <line x1="21" y1="21" x2="16.65" y2="16.65" />
                <line x1="11" y1="8" x2="11" y2="14" />
                <line x1="8" y1="11" x2="14" y2="11" />
              </svg>
              Analyze Surface
            </>
          )}
        </button>

        <div className="defect-chips">
          <span className="chips-label">Detectable defects</span>
          <div className="chips-list">
            {['Crazing', 'Inclusion', 'Patches', 'Pitted Surface', 'Rolled-in Scale', 'Scratches'].map(d => (
              <span key={d} className="chip">{d}</span>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
