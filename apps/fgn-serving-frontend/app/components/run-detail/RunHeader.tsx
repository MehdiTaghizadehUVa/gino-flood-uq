import type { ReactNode } from "react";
import { PageHeader } from "../PageHeader";
import { StatusBadge, statusTone } from "../StatusBadge";

type RunHeaderProps = {
  title: string;
  status?: string | null;
  createdAt?: string | null;
  pinned?: boolean;
  cache?: {
    materialized_from_cache?: boolean;
    waiting_for_cached_result?: boolean;
  } | null;
  actions?: ReactNode;
};

export function RunHeader({ title, status, createdAt, pinned, cache, actions }: RunHeaderProps) {
  return (
    <div className="run-command-header">
      <PageHeader
        kicker="Run detail console"
        title={title}
        subtitle={
          status ? (
            <>
              <StatusBadge tone={statusTone(status)}>{status}</StatusBadge>
              {cache?.materialized_from_cache ? (
                <StatusBadge tone="success">Loaded from verified cache</StatusBadge>
              ) : null}
              {cache?.waiting_for_cached_result ? (
                <StatusBadge tone="active">Waiting for matching run</StatusBadge>
              ) : null}
              {createdAt ? <span> created {new Date(createdAt).toLocaleString()}</span> : null}
              {pinned ? <span> · pinned</span> : null}
            </>
          ) : (
            "Loading run metadata, artifacts, and monitoring reports."
          )
        }
        actions={actions}
      />
    </div>
  );
}
