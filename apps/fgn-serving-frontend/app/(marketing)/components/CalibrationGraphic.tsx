import { calibrationCurve } from "../content";
import { calibrationLayerState } from "./scrollSceneMath.mjs";

const WIDTH = 620;
const HEIGHT = 420;
const PAD = 58;
const x = (value: number) => PAD + value * (WIDTH - PAD * 2);
const y = (value: number) => HEIGHT - PAD - value * (HEIGHT - PAD * 2);
const curvePoints = calibrationCurve.map(([raw, calibrated]) => `${x(raw)},${y(calibrated)}`).join(" ");

export function CalibrationGraphic({ stage = 2 }: { stage?: number }) {
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  const normalizedStage = Math.min(2, Math.max(0, stage));
  const layerState = calibrationLayerState(normalizedStage);
  return (
    <figure
      className={`calibration-figure calibration-stage-${normalizedStage}`}
      data-points-visible={layerState.pointsVisible}
      data-curve-visible={layerState.curveVisible}
    >
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-labelledby="calibration-title calibration-desc">
        <title id="calibration-title">Isotonic probability mapping for a 0.30 meter exceedance threshold</title>
        <desc id="calibration-desc">Raw ensemble probability on the horizontal axis and calibrated probability on the vertical axis.</desc>
        {ticks.map((tick) => (
          <g key={tick}>
            <line className="chart-grid" x1={x(tick)} x2={x(tick)} y1={y(0)} y2={y(1)} />
            <line className="chart-grid" x1={x(0)} x2={x(1)} y1={y(tick)} y2={y(tick)} />
            <text className="chart-tick" x={x(tick)} y={HEIGHT - 20} textAnchor="middle">{tick.toFixed(2)}</text>
            <text className="chart-tick" x={30} y={y(tick) + 4} textAnchor="middle">{tick.toFixed(2)}</text>
          </g>
        ))}
        <line className="chart-reference" x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} />
        <polyline className="chart-curve" points={curvePoints} pathLength="1" />
        {calibrationCurve.filter((_, index) => index % 2 === 0).map(([raw, calibrated]) => (
          <circle key={raw} className="chart-point" cx={x(raw)} cy={y(calibrated)} r="4" />
        ))}
        <text className="chart-axis-label" x={WIDTH / 2} y={HEIGHT - 1} textAnchor="middle">Raw ensemble probability</text>
        <text className="chart-axis-label" transform={`translate(14 ${HEIGHT / 2}) rotate(-90)`} textAnchor="middle">Calibrated probability</text>
      </svg>
      <figcaption>
        The mapping is explicit and versioned. Raw probabilities remain available beside calibrated products for audit.
      </figcaption>
    </figure>
  );
}
