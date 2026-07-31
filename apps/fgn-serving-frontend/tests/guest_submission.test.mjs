import assert from "node:assert/strict";
import test from "node:test";

import {
  GUEST_SUBMISSION_DRAFT_VERSION,
  MAX_GUEST_DRAFT_CSV_CHARS,
  buildSubmissionSignInPath,
  parseGuestSubmissionDraft,
} from "../app/(demo)/demo/guestSubmission.mjs";

test("submission sign-in returns to the configured demo draft", () => {
  assert.equal(
    buildSubmissionSignInPath(),
    "/oauth2/start?rd=%2Fdemo%3Fworkspace%3Dnew%26resume%3Dsubmit",
  );
});

test("guest submission draft restores only the bounded scientific form", () => {
  const restored = parseGuestSubmissionDraft(
    JSON.stringify({
      version: GUEST_SUBMISSION_DRAFT_VERSION,
      fileName: "scenario.csv",
      fileType: "text/csv",
      csv: "time_seconds,stage,precipitation\n0,1.2,0\n",
      label: "Guest scenario",
      forecastSteps: "94",
      thresholds: "0.05,0.3",
      ensembleCount: "3",
      membersPerEnsemble: "20",
      requestAnimation: false,
      userEmail: "must-not-be-restored@example.com",
    }),
  );

  assert.deepEqual(restored, {
    version: GUEST_SUBMISSION_DRAFT_VERSION,
    fileName: "scenario.csv",
    fileType: "text/csv",
    csv: "time_seconds,stage,precipitation\n0,1.2,0\n",
    label: "Guest scenario",
    forecastSteps: "94",
    thresholds: "0.05,0.3",
    ensembleCount: "3",
    membersPerEnsemble: "20",
    requestAnimation: false,
  });
});

test("guest submission draft rejects malformed and oversized payloads", () => {
  assert.equal(parseGuestSubmissionDraft("not json"), null);
  assert.equal(
    parseGuestSubmissionDraft(
      JSON.stringify({
        version: GUEST_SUBMISSION_DRAFT_VERSION,
        fileName: "too-large.csv",
        csv: "x".repeat(MAX_GUEST_DRAFT_CSV_CHARS + 1),
      }),
    ),
    null,
  );
});
