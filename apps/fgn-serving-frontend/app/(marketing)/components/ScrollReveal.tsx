"use client";

import { useEffect } from "react";

export function ScrollReveal() {
  useEffect(() => {
    const nodes = Array.from(document.querySelectorAll<HTMLElement>("[data-reveal]"));
    document.body.setAttribute("data-reveal-ready", "true");
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          (entry.target as HTMLElement).setAttribute("data-reveal-visible", "true");
          observer.unobserve(entry.target);
        });
      },
      // Long evidence sections can be many viewports tall. A small threshold
      // keeps the reveal reachable while still waiting for visible content.
      { rootMargin: "0px 0px -10% 0px", threshold: 0.01 }
    );
    nodes.forEach((node) => observer.observe(node));
    return () => observer.disconnect();
  }, []);

  return null;
}
