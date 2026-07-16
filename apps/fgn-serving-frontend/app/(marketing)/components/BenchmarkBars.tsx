type RuntimeBar = {
  label: string;
  value: number;
  display: string;
  tone: string;
};

export function BenchmarkBars({
  bars,
  max,
  ariaLabel = "Measured benchmark values"
}: {
  bars: readonly RuntimeBar[];
  max: number;
  ariaLabel?: string;
}) {
  return (
    <div className="benchmark-bars" aria-label={ariaLabel}>
      {bars.map((bar) => (
        <div className="benchmark-row" key={bar.label}>
          <div className="benchmark-label">
            <span>{bar.label}</span>
            <strong>{bar.display}</strong>
          </div>
          <div className="benchmark-track" aria-hidden="true">
            <span className={`benchmark-fill ${bar.tone}`} style={{ width: `${Math.max(4, (bar.value / max) * 100)}%` }} />
          </div>
        </div>
      ))}
    </div>
  );
}
