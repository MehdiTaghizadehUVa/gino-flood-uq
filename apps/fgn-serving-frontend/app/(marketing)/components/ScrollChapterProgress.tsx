"use client";

import { useEffect, useRef, useState } from "react";

const CHAPTERS = [
  { id: "top", label: "Overview" },
  { id: "capabilities", label: "Decision context" },
  { id: "calibration", label: "Calibration" },
  { id: "uncertainty", label: "Uncertainty" },
  { id: "monitoring", label: "Monitoring" },
  { id: "evidence", label: "Portsmouth proof" }
] as const;

export function ScrollChapterProgress() {
  const [activeId, setActiveId] = useState<string>(CHAPTERS[0].id);
  const activeIdRef = useRef(activeId);

  useEffect(() => {
    let frameRequest = 0;
    const sections = CHAPTERS.map((chapter) => document.getElementById(chapter.id)).filter(
      (section): section is HTMLElement => Boolean(section)
    );

    const measure = () => {
      frameRequest = 0;
      const targetLine = window.innerHeight * 0.42;
      let nextId = sections[0]?.id ?? CHAPTERS[0].id;
      for (const section of sections) {
        const bounds = section.getBoundingClientRect();
        if (bounds.top <= targetLine) nextId = section.id;
        if (bounds.top > targetLine) break;
      }
      if (nextId !== activeIdRef.current) {
        activeIdRef.current = nextId;
        setActiveId(nextId);
      }
    };

    const scheduleMeasure = () => {
      if (!frameRequest) frameRequest = window.requestAnimationFrame(measure);
    };
    scheduleMeasure();
    window.addEventListener("scroll", scheduleMeasure, { passive: true });
    window.addEventListener("resize", scheduleMeasure);
    return () => {
      window.cancelAnimationFrame(frameRequest);
      window.removeEventListener("scroll", scheduleMeasure);
      window.removeEventListener("resize", scheduleMeasure);
    };
  }, []);

  return (
    <nav className="scroll-chapter-progress" aria-label="FloodUQ story chapters">
      <ol>
        {CHAPTERS.map((chapter, index) => (
          <li key={chapter.id} className={activeId === chapter.id ? "active" : ""}>
            <a href={`#${chapter.id}`} aria-current={activeId === chapter.id ? "location" : undefined}>
              <span>{String(index + 1).padStart(2, "0")}</span>
              <strong>{chapter.label}</strong>
            </a>
          </li>
        ))}
      </ol>
    </nav>
  );
}
