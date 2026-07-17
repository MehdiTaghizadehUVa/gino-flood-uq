"use client";

import { useEffect } from "react";

const MOBILE_REVEAL_QUERY = "(max-width: 899px)";
const MOBILE_SECTION_ROOT_MARGIN = "0px 0px -22% 0px";
const MOBILE_ITEM_ROOT_MARGIN = "0px 0px -28% 0px";
const MOBILE_REVEAL_SELECTOR = [
  ".scroll-story-steps > li",
  ".numbered-grid > .numbered-item",
  ".service-lifecycle > li",
  ".product-gallery > figure",
  ".comparison-matrix tbody > tr",
  ".case-study-stats > div",
  ".case-study-chapter"
].join(",");

export function ScrollReveal() {
  useEffect(() => {
    const nodes = Array.from(document.querySelectorAll<HTMLElement>("[data-reveal]"));
    const isMobile = window.matchMedia(MOBILE_REVEAL_QUERY).matches;
    const mobileNodes = isMobile
      ? Array.from(document.querySelectorAll<HTMLElement>(MOBILE_REVEAL_SELECTOR))
      : [];

    mobileNodes.forEach((node, index) => {
      node.setAttribute("data-mobile-reveal", "true");
      node.style.setProperty("--mobile-reveal-delay", `${(index % 4) * 55}ms`);
    });

    if (!("IntersectionObserver" in window)) {
      nodes.forEach((node) => node.setAttribute("data-reveal-visible", "true"));
      mobileNodes.forEach((node) => node.setAttribute("data-mobile-reveal-visible", "true"));
      document.body.setAttribute("data-reveal-ready", "true");
      return () => {
        document.body.removeAttribute("data-reveal-ready");
      };
    }

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
      { rootMargin: isMobile ? MOBILE_SECTION_ROOT_MARGIN : "0px 0px -10% 0px", threshold: 0.01 }
    );
    const mobileObserver = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          (entry.target as HTMLElement).setAttribute("data-mobile-reveal-visible", "true");
          mobileObserver.unobserve(entry.target);
        });
      },
      { rootMargin: MOBILE_ITEM_ROOT_MARGIN, threshold: 0.01 }
    );

    nodes.forEach((node) => observer.observe(node));
    mobileNodes.forEach((node) => mobileObserver.observe(node));
    document.body.setAttribute("data-reveal-ready", "true");

    return () => {
      observer.disconnect();
      mobileObserver.disconnect();
      mobileNodes.forEach((node) => node.style.removeProperty("--mobile-reveal-delay"));
      document.body.removeAttribute("data-reveal-ready");
    };
  }, []);

  return null;
}
