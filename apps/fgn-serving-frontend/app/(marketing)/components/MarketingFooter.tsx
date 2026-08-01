import Image from "next/image";
import { ArrowUpRight } from "lucide-react";
import { COLLABORATION_MAILTO } from "../content";

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
          <strong>Coastal flood scenarios you can compare, explain, and improve.</strong>
          <p>A managed service for flood probability and forecast agreement, developed at the University of Virginia.</p>
        </div>
        <div className="footer-links">
          <a href={COLLABORATION_MAILTO}>Discuss a collaboration <ArrowUpRight size={14} /></a>
          <a href="/demo">Portsmouth demo</a>
        </div>
      </div>
      <div className="footer-legal">
        <span>Fast coastal scenario analysis with calibrated probabilities and traceable evidence.</span>
        <span>© 2026 FloodUQ</span>
      </div>
    </footer>
  );
}
