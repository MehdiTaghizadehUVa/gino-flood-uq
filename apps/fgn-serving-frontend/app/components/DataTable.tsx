import type { ReactNode } from "react";

type DataTableProps = {
  children: ReactNode;
  className?: string;
};

export function DataTable({ children, className = "" }: DataTableProps) {
  return (
    <div className={`data-table-wrap ${className}`.trim()}>
      <table className="data-table">{children}</table>
    </div>
  );
}
