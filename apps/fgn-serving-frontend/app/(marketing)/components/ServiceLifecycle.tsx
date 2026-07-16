import type { NumberedContentItem } from "../content";

export function ServiceLifecycle({
  items,
  ariaLabel
}: {
  items: readonly NumberedContentItem[];
  ariaLabel: string;
}) {
  return (
    <ol className="service-lifecycle" aria-label={ariaLabel}>
      {items.map((item) => (
        <li key={`${item.number}-${item.title}`}>
          <span className="lifecycle-number">{item.number}</span>
          <div>
            <h3>{item.title}</h3>
            <p>{item.body}</p>
          </div>
        </li>
      ))}
    </ol>
  );
}
