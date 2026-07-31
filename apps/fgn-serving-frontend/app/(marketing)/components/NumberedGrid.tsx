type NumberedItem = { number: string; title: string; body: string };

export function NumberedGrid({ items, columns = 3 }: { items: readonly NumberedItem[]; columns?: 2 | 3 | 4 }) {
  return (
    <div className={`numbered-grid columns-${columns}`}>
      {items.map((item) => (
        <article className="numbered-item" key={`${item.number}-${item.title}`}>
          <span>{item.number}</span>
          <h3>{item.title}</h3>
          <p>{item.body}</p>
        </article>
      ))}
    </div>
  );
}
