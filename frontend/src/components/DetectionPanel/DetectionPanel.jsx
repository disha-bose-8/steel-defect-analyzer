import './DetectionPanel.css'

function ConfidencePill({ confidence }) {
  const pct = Math.round(confidence * 100)
  const cls = pct >= 80 ? 'pill--high' : pct >= 50 ? 'pill--mid' : 'pill--low'
  return <span className={`conf-pill ${cls}`}>{pct}%</span>
}

export default function DetectionPanel({
  result,
  error,
  unavailable,
  originalImageUrl,
  isAnalyzing,
}) {
  const showEmpty = !result && !error && !unavailable && !isAnalyzing

  return (
    <div className="panel detection-panel">
      <div className="panel-header">
        <svg className="panel-header-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
          <rect x="3" y="3" width="18" height="18" rx="2" />
          <path d="M3 9h18M9 21V9" />
        </svg>
        <span className="panel-title">Object Detection</span>
        {result && (
          <span className="detection-badge">
            {result.num_detections} detection{result.num_detections !== 1 ? 's' : ''}
          </span>
        )}
      </div>

      <div className="detection-body">
        {isAnalyzing && !result && (
          <div className="detection-state">
            <div className="skeleton-image" />
            <div className="skeleton-rows">
              <div className="skeleton-row" style={{ width: '70%' }} />
              <div className="skeleton-row" style={{ width: '55%' }} />
              <div className="skeleton-row" style={{ width: '40%' }} />
            </div>
          </div>
        )}

        {unavailable && (
          <div className="detection-state detection-state--unavailable">
            <div className="unavailable-icon" aria-hidden="true">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M12 2a10 10 0 1 0 0 20A10 10 0 0 0 12 2z" />
                <path d="M12 8v4M12 16h.01" />
              </svg>
            </div>
            <p className="unavailable-title">Detector training in progress</p>
            <p className="unavailable-sub">The YOLOv8 detection model is still being trained. Classification results are still available below.</p>
          </div>
        )}

        {error && (
          <div className="detection-state detection-state--error">
            <p className="error-msg">{error}</p>
          </div>
        )}

        {showEmpty && (
          <div className="detection-state detection-state--empty">
            {originalImageUrl ? (
              <>
                <img src={originalImageUrl} alt="Uploaded preview" className="preview-idle" />
                <p className="idle-hint">Hit Analyze to run detection</p>
              </>
            ) : (
              <p className="idle-hint">Upload an image to get started</p>
            )}
          </div>
        )}

        {result && (
  <div className="detection-results">
    <div className="annotated-image-container">
      <img
        src={`data:image/png;base64,${result.annotated_image_b64}`}
        alt="Detection overlay"
        className="result-img-full"
      />
    </div>

    <div className="detection-summary">
      <span className="summary-msg">{result.message}</span>
    </div>


  </div>
)}
      </div>
    </div>
  )
}
