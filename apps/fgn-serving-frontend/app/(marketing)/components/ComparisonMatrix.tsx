import type { ComparisonMethod, ComparisonRow } from "../content";

type ComparisonMatrixProps = {
  methods: readonly ComparisonMethod[];
  rows: readonly ComparisonRow[];
};

export function ComparisonMatrix({ methods, rows }: ComparisonMatrixProps) {
  return (
    <div className="comparison-matrix-wrap">
      <table className="comparison-matrix">
        <caption>
          Method roles and capabilities. The comparison describes how the methods are used in this service context;
          measured performance claims are limited to the Portsmouth evidence section.
        </caption>
        <thead>
          <tr>
            <th scope="col">Capability</th>
            {methods.map((method) => (
              <th className={method.id === "flooduq" ? "featured" : ""} key={method.id} scope="col">
                <strong>{method.label}</strong>
                <span>{method.qualifier}</span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.dimension}>
              <th scope="row">{row.dimension}</th>
              {methods.map((method) => (
                <td className={method.id === "flooduq" ? "featured" : ""} key={method.id}>
                  <span className="mobile-method-label">{method.label}</span>
                  {row[method.id]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
