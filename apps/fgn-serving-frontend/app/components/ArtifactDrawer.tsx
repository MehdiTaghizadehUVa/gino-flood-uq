import { Database, Download, FileText, Film, FolderOpen, Image } from "lucide-react";

type Artifact = {
  artifact_id?: string;
  id?: string;
  filename?: string;
  name?: string;
  label?: string;
  size_bytes?: number | null;
  content_type?: string | null;
};

type ArtifactDrawerProps = {
  artifacts: Artifact[];
  hrefForArtifact: (artifact: Artifact) => string;
  title?: string;
  initiallyOpen?: boolean;
};

function formatBytes(value?: number | null) {
  if (!value || value <= 0) return "";
  const units = ["B", "KB", "MB", "GB"];
  let size = value;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

function artifactName(artifact: Artifact) {
  return artifact.label || artifact.filename || artifact.name || artifact.artifact_id || artifact.id || "artifact";
}

function artifactKey(artifact: Artifact) {
  return artifact.artifact_id || artifact.id || artifactName(artifact);
}

function groupName(artifact: Artifact) {
  const name = artifactName(artifact).toLowerCase();
  const type = (artifact.content_type || "").toLowerCase();
  if (type.includes("json") || name.endsWith(".json")) return "JSON summaries";
  if (type.includes("image") || /\.(png|jpg|jpeg|svg)$/i.test(name)) return "Figures and maps";
  if (type.includes("gif") || type.includes("video") || /\.(gif|mp4)$/i.test(name)) return "Animations";
  if (name.endsWith(".h5") || name.endsWith(".hdf5")) return "Ensemble data";
  return "Run files";
}

function groupIcon(group: string) {
  if (group === "JSON summaries") return <FileText size={15} />;
  if (group === "Figures and maps") return <Image size={15} />;
  if (group === "Animations") return <Film size={15} />;
  if (group === "Ensemble data") return <Database size={15} />;
  return <FolderOpen size={15} />;
}

export function ArtifactDrawer({
  artifacts,
  hrefForArtifact,
  title = "Downloads",
  initiallyOpen = false
}: ArtifactDrawerProps) {
  const groups = artifacts.reduce<Record<string, Artifact[]>>((acc, artifact) => {
    const group = groupName(artifact);
    acc[group] = acc[group] || [];
    acc[group].push(artifact);
    return acc;
  }, {});
  const groupEntries = Object.entries(groups);

  return (
    <details className="artifact-drawer" open={initiallyOpen}>
      <summary>
        <span>{title}</span>
        <span className="metric-card-detail">{artifacts.length} artifacts</span>
      </summary>
      {groupEntries.length === 0 ? (
        <div className="artifact-group">
          <p className="section-subtitle">Artifacts will appear after postprocessing completes.</p>
        </div>
      ) : (
        groupEntries.map(([group, items]) => (
          <section key={group} className="artifact-group">
            <div className="metric-card-header">
              <span>
                {groupIcon(group)} {group}
              </span>
              <span>{items.length}</span>
            </div>
            <div className="artifact-list">
              {items.map((artifact) => (
                <a
                  key={artifactKey(artifact)}
                  className="artifact-row"
                  href={hrefForArtifact(artifact)}
                  target="_blank"
                  rel="noreferrer"
                >
                  <span>{artifactName(artifact)}</span>
                  <small>
                    {formatBytes(artifact.size_bytes)}
                    <Download size={13} aria-hidden="true" />
                  </small>
                </a>
              ))}
            </div>
          </section>
        ))
      )}
    </details>
  );
}
