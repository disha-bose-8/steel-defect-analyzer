import './StatusBar.css'

export default function StatusBar({ health }) {
  if (!health) {
    return (
      <div className="status-bar">
        <span className="status-dot status-dot--unknown" />
        <span className="status-label">Connecting…</span>
      </div>
    )
  }

  const classifierOk = health.classifier_loaded
  const detectorOk = health.detector_loaded

  return (
    <div className="status-bar">
      <div className="status-item">
        <span className={`status-dot ${classifierOk ? 'status-dot--ok' : 'status-dot--err'}`} />
        <span className="status-label">Classifier</span>
      </div>
      <div className="status-divider" />
      <div className="status-item">
        <span className={`status-dot ${detectorOk ? 'status-dot--ok' : 'status-dot--warn'}`} />
        <span className="status-label">Detector</span>
        {!detectorOk && <span className="status-badge">Training</span>}
      </div>
    </div>
  )
}
