const CASE_STUDY_ASSET_RELEASE = "20260717-title-free";

export function caseStudyAsset(src: string): string {
  const separator = src.includes("?") ? "&" : "?";
  return `${src}${separator}v=${CASE_STUDY_ASSET_RELEASE}`;
}
