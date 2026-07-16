import { ArrowUpRight } from "lucide-react";
import { PILOT_MAILTO, RESEARCH_DISCLAIMER } from "../content";

export function MarketingFooter() {
  return (
    <footer className="marketing-footer">
      <div className="footer-wordmark" aria-label="FloodUQ">
        {"FLOODUQ".split("").map((letter, index) => <span key={`${letter}-${index}`}>{letter}</span>)}
      </div>
      <div className="footer-grid">
        <div>
          <strong>Managed coastal flood uncertainty, made inspectable.</strong>
          <p>A domain-specific modeling service developed at the University of Virginia.</p>
        </div>
        <div className="footer-links">
          <a href={PILOT_MAILTO}>Request a pilot <ArrowUpRight size={14} /></a>
          <a href="/demo">Portsmouth demo</a>
        </div>
      </div>
      <div className="footer-legal">
        <span>{RESEARCH_DISCLAIMER}</span>
        <span>© 2026 FloodUQ</span>
      </div>
    </footer>
  );
}
