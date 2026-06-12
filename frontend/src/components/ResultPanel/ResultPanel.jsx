import './ResultPanel.css'

const DEFECT_COLORS = {
  crazing: '#7b8cde',
  inclusion: '#5cb8a0',
  patches: '#d4a843',
  pitted_surface: '#e05c6a',
  'rolled-in_scale': '#a8b5c8',
  scratches: '#c070d4',
}

function getDefectColor(name) {
  const key = name?.toLowerCase().replace(/\s+/g, '_')
  return DEFECT_COLORS[key] || '#7b8cde'
}

function ConfidenceBar({ value, color }) {
  return (
    <div className="conf-bar-track" role="meter" aria-valuenow={value} aria-valuemin={0} aria-valuemax={100}>
      <div
        className="conf-bar-fill"
        style={{ width: `${value}%`, background: color }}
      />
    </div>
  )
}

export default function ResultPanel({ result, error, isAnalyzing }) {
  if (isAnalyzing && !result) {
    return (
      <div className="panel result-panel">
        <div className="panel-header">
          <svg className="panel-header-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
          </svg>
          <span className="panel-title">Classification</span>
        </div>
        <div className="result-body">
          <div className="result-skeleton">
            <div className="skel-block" style={{ height: 28, width: '60%' }} />
            <div className="skel-block" style={{ height: 14, width: '40%' }} />
            <div className="skel-block" style={{ height: 8, marginTop: 8 }} />
          </div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="panel result-panel">
        <div className="panel-header">
          <svg className="panel-header-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
          </svg>
          <span className="panel-title">Classification</span>
        </div>
        <div className="result-body">
          <div className="result-error">{error}</div>
        </div>
      </div>
    )
  }

  if (!result) return null

  const primaryColor = getDefectColor(result.predicted_defect)

  return (
    <div className="panel result-panel">
      <div className="panel-header">
        <svg className="panel-header-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
          <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
        </svg>
        <span className="panel-title">Classification</span>
        <span className="result-status-chip">ResNet-18</span>
      </div>

      <div className="result-body">
        <div className="primary-result">
          <div className="defect-dot" style={{ background: primaryColor }} aria-hidden="true" />
          <div className="primary-info">
            <h2 className="defect-name">{result.predicted_defect.replace(/_/g, ' ')}</h2>
            <div className="confidence-row">
              <span className="confidence-value" style={{ color: primaryColor }}>
                {result.confidence.toFixed(1)}%
              </span>
              <span className="confidence-label">confidence</span>
            </div>
          </div>
        </div>

        <div className="primary-bar-wrap">
          <ConfidenceBar value={result.confidence} color={primaryColor} />
        </div>

        {result.description && (
          <p className="defect-description">{result.description}</p>
        )}

        <div className="top3-section">
          <div className="top3-header">
            <span className="top3-label">Top predictions</span>
          </div>
          <div className="top3-list">
            {result.top3.map((item, i) => {
              const color = getDefectColor(item.defect)
              return (
                <div key={i} className={`top3-row ${i === 0 ? 'top3-row--first' : ''}`}>
                  <span className="top3-rank">#{i + 1}</span>
                  <span className="top3-name">{item.defect.replace(/_/g, ' ')}</span>
                  <div className="top3-bar-wrap">
                    <ConfidenceBar value={item.confidence} color={color} />
                  </div>
                  <span className="top3-pct" style={{ color: i === 0 ? color : undefined }}>
                    {item.confidence.toFixed(1)}%
                  </span>
                </div>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}
