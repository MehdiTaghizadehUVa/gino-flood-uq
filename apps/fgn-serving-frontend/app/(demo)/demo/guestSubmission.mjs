export const GUEST_SUBMISSION_DRAFT_KEY = "flooduq.guest-submission.v1";
export const GUEST_SUBMISSION_DRAFT_VERSION = 1;
export const MAX_GUEST_DRAFT_CSV_CHARS = 2 * 1024 * 1024;

export function buildSubmissionSignInPath(
  returnPath = "/demo?workspace=new&resume=submit",
) {
  const params = new URLSearchParams({ rd: returnPath });
  return `/oauth2/start?${params.toString()}`;
}

export function buildAnalysisQueueSignInPath(
  returnPath = "/demo?workspace=runs",
) {
  const params = new URLSearchParams({ rd: returnPath });
  return `/oauth2/start?${params.toString()}`;
}

export function parseGuestSubmissionDraft(raw) {
  if (typeof raw !== "string" || !raw) return null;
  try {
    const draft = JSON.parse(raw);
    if (
      draft?.version !== GUEST_SUBMISSION_DRAFT_VERSION ||
      typeof draft.fileName !== "string" ||
      typeof draft.csv !== "string" ||
      draft.csv.length === 0 ||
      draft.csv.length > MAX_GUEST_DRAFT_CSV_CHARS
    ) {
      return null;
    }
    return {
      version: GUEST_SUBMISSION_DRAFT_VERSION,
      fileName: draft.fileName,
      fileType: typeof draft.fileType === "string" ? draft.fileType : "text/csv",
      csv: draft.csv,
      label: typeof draft.label === "string" ? draft.label : "",
      forecastSteps: typeof draft.forecastSteps === "string" ? draft.forecastSteps : "",
      thresholds: typeof draft.thresholds === "string" ? draft.thresholds : "0.01,0.05,0.1,0.3,0.5",
      ensembleCount: typeof draft.ensembleCount === "string" ? draft.ensembleCount : "",
      membersPerEnsemble:
        typeof draft.membersPerEnsemble === "string" ? draft.membersPerEnsemble : "",
      requestAnimation: draft.requestAnimation !== false,
    };
  } catch {
    return null;
  }
}
