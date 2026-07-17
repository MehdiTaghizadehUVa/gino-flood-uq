import Image from "next/image";
import { ArrowUpRight } from "lucide-react";
import { marketingNav, PILOT_MAILTO } from "../content";

export function MarketingNav() {
  return (
    <header className="marketing-nav">
      <a className="marketing-brand" href="#top" aria-label="FloodUQ home">
        <Image
          className="marketing-brand-lockup"
          src="/marketing/brand/flooduq-logo-lockup.png"
          alt=""
          width={1453}
          height={327}
          priority
        />
      </a>
      <nav className="marketing-nav-links" aria-label="Product navigation">
        {marketingNav.map((item) => (
          <a key={item.href} href={item.href}>{item.label}</a>
        ))}
      </nav>
      <div className="marketing-nav-actions">
        <a className="text-link" href="/demo">
          Portsmouth demo
        </a>
        <a className="primary-cta compact" href={PILOT_MAILTO}>
          Request a pilot <ArrowUpRight size={15} aria-hidden="true" />
        </a>
      </div>
      <details className="marketing-mobile-menu">
        <summary aria-label="Open navigation">Menu</summary>
        <nav aria-label="Mobile product navigation">
          {marketingNav.map((item) => (
            <a key={item.href} href={item.href}>{item.label}</a>
          ))}
          <a href="/demo">Portsmouth demo</a>
          <a href={PILOT_MAILTO}>Request a pilot</a>
        </nav>
      </details>
    </header>
  );
}
