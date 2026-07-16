"use client";

import { Pause, Play, RotateCcw, SkipBack, SkipForward } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import type { CaseStudyProduct } from "./caseStudyTypes";
import { EvidenceCaption } from "./EvidenceCaption";

const SPEEDS = [
  { label: "0.5×", delay: 800 },
  { label: "1×", delay: 450 },
  { label: "2×", delay: 240 }
] as const;

export function ForecastStoryPlayer({
  eventLabel,
  products,
  posterSrc
}: {
  eventLabel: string;
  products: CaseStudyProduct[];
  posterSrc: string;
}) {
  const rootRef = useRef<HTMLDivElement>(null);
  const [productId, setProductId] = useState(products[0]?.id ?? "probability");
  const [frameIndex, setFrameIndex] = useState(0);
  const [speedIndex, setSpeedIndex] = useState(1);
  const [playing, setPlaying] = useState(false);
  const [visible, setVisible] = useState(true);
  const product = useMemo(
    () => products.find((item) => item.id === productId) ?? products[0],
    [productId, products]
  );
  const frame = product?.frames[frameIndex] ?? product?.frames[0];

  useEffect(() => {
    const query = window.matchMedia("(prefers-reduced-motion: reduce)");
    if (!query.matches) setPlaying(true);
    const node = rootRef.current;
    if (!node || !("IntersectionObserver" in window)) return;
    const observer = new IntersectionObserver(([entry]) => setVisible(entry.isIntersecting), { threshold: 0.18 });
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!playing || !visible || !product?.frames.length) return;
    const timer = window.setInterval(
      () => setFrameIndex((current) => (current + 1) % product.frames.length),
      SPEEDS[speedIndex].delay
    );
    return () => window.clearInterval(timer);
  }, [playing, product, speedIndex, visible]);

  useEffect(() => {
    if (!product?.frames.length) return;
    for (const offset of [1, 2]) {
      const next = product.frames[(frameIndex + offset) % product.frames.length];
      const image = new window.Image();
      image.src = next.src;
    }
  }, [frameIndex, product]);

  if (!product || !frame) return null;
  const step = (delta: number) => {
    setPlaying(false);
    setFrameIndex((current) => (current + delta + product.frames.length) % product.frames.length);
  };

  return (
    <div ref={rootRef} className="forecast-story-player">
      <div className="forecast-player-toolbar">
        <div className="case-study-segmented" aria-label="Forecast product">
          {products.map((item) => (
            <button
              type="button"
              key={item.id}
              className={item.id === product.id ? "active" : ""}
              aria-pressed={item.id === product.id}
              onClick={() => {
                setProductId(item.id);
                setFrameIndex(0);
              }}
            >
              {item.label}
            </button>
          ))}
        </div>
        <span className="forecast-lead" aria-live="polite">Lead +{frame.leadHours.toFixed(2)} h</span>
      </div>

      <figure className="case-study-figure forecast-player-figure">
        <div className="forecast-player-canvas">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={frame.src || posterSrc}
            alt={`${eventLabel} ${product.label} at ${frame.leadHours.toFixed(2)} forecast hours`}
            width={1400}
            height={1080}
            loading="lazy"
          />
        </div>
        <div className="forecast-transport" aria-label="Forecast animation controls">
          <button type="button" onClick={() => step(-1)} title="Previous frame" aria-label="Previous frame"><SkipBack size={17} /></button>
          <button
            type="button"
            className="transport-primary"
            onClick={() => setPlaying((current) => !current)}
            title={playing ? "Pause" : "Play"}
            aria-label={playing ? "Pause forecast animation" : "Play forecast animation"}
          >
            {playing ? <Pause size={18} /> : <Play size={18} />}
          </button>
          <button type="button" onClick={() => step(1)} title="Next frame" aria-label="Next frame"><SkipForward size={17} /></button>
          <input
            type="range"
            min={0}
            max={product.frames.length - 1}
            step={1}
            value={frameIndex}
            aria-label={`Forecast lead time, ${frame.leadHours.toFixed(2)} hours`}
            onChange={(event) => {
              setPlaying(false);
              setFrameIndex(Number(event.target.value));
            }}
          />
          <button
            type="button"
            onClick={() => {
              setFrameIndex(0);
              setPlaying(true);
            }}
            title="Replay"
            aria-label="Replay forecast"
          ><RotateCcw size={16} /></button>
          <label>
            <span className="sr-only">Playback speed</span>
            <select value={speedIndex} onChange={(event) => setSpeedIndex(Number(event.target.value))}>
              {SPEEDS.map((speed, index) => <option key={speed.label} value={index}>{speed.label}</option>)}
            </select>
          </label>
        </div>
        <EvidenceCaption
          title="A forecast through time, not a single peak map"
          insight="Track where the selected product emerges, concentrates, and recedes across the full Irene 2011 horizon."
          method="Frames are sampled from the complete production rollout while preserving onset, peak footprint, peak disagreement, and recession."
        />
      </figure>
    </div>
  );
}
