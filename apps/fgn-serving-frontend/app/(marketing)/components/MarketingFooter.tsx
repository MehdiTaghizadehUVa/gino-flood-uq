import Image from "next/image";
import { ArrowUpRight } from "lucide-react";
import { PILOT_MAILTO } from "../content";

export function MarketingFooter() {
  return (
    <footer className="marketing-footer">
      <div className="footer-logo" aria-label="FloodUQ">
        <Image
          src="/marketing/brand/flooduq-logo-lockup.png"
          alt=""
          width={1453}
          height={327}
          sizes="(max-width: 760px) calc(100vw - 36px), 1100px"
        />
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
        <span>Domain-specific coastal flood intelligence with documented validation and provenance.</span>
        <span>© 2026 FloodUQ</span>
      </div>
    </footer>
  );
}
