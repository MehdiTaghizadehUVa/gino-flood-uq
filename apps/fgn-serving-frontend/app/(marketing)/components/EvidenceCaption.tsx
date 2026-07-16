import { Info } from "lucide-react";

export function EvidenceCaption({
  title,
  insight,
  method
}: {
  title: string;
  insight: string;
  method: string;
}) {
  return (
    <figcaption className="case-study-caption">
      <div>
        <strong>{title}</strong>
        <p>{insight}</p>
      </div>
      <details className="case-study-method">
        <summary aria-label={`Method note for ${title}`} title="Method and limitations">
          <Info size={15} aria-hidden="true" />
        </summary>
        <p>{method}</p>
      </details>
    </figcaption>
  );
}
