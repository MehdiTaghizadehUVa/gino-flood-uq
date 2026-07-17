"use client";

import { Pause, Play, RotateCcw, SkipBack, SkipForward } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { caseStudyAsset } from "../caseStudyAsset";
import type { CaseStudyProduct } from "./caseStudyTypes";
import { EvidenceCaption } from "./EvidenceCaption";
import { depthStoryMilestoneFrames, milestoneIndexFromFrame } from "./scrollSceneMath.mjs";

const SPEEDS = [
  { label: "0.5×", rate: 0.5 },
  { label: "1×", rate: 1 },
  { label: "1.5×", rate: 1.5 }
] as const;

export function ForecastStoryPlayer({
  eventLabel,
  products,
  posterSrc,
  peakMeanDepthTimeIndex
}: {
  eventLabel: string;
  products: CaseStudyProduct[];
  posterSrc: string;
  peakMeanDepthTimeIndex: number;
}) {
  const rootRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [productId, setProductId] = useState(
    () => products.find((item) => item.id === "meanDepth")?.id ?? products[0]?.id ?? "meanDepth"
  );
  const [frameIndex, setFrameIndex] = useState(0);
  const [speedIndex, setSpeedIndex] = useState(1);
  const [playing, setPlaying] = useState(true);
  const [visible, setVisible] = useState(false);
  const product = useMemo(
    () => products.find((item) => item.id === productId) ?? products[0],
    [productId, products]
  );
  const frame = product?.frames[frameIndex] ?? product?.frames[0];
  const milestones = useMemo(() => {
    const frames = products.find((item) => item.id === "meanDepth")?.frames ?? products[0]?.frames ?? [];
    const lastIndex = Math.max(0, frames.length - 1);
    const indexForTime = (timeIndex: number) => {
      const index = frames.findIndex((item) => item.timeIndex === timeIndex);
      return index >= 0 ? index : 0;
    };
    const [initial, earlyResponse, inlandExpansion, peakDepth, recession] = depthStoryMilestoneFrames(
      lastIndex,
      indexForTime(peakMeanDepthTimeIndex)
    );
    return [
      { label: "Initial mean-depth field", body: "The forecast opens from its forcing-conditioned water-depth history.", frameIndex: initial },
      { label: "Coastal depth response", body: "Mean water depth begins increasing along the connected coastal response pathways.", frameIndex: earlyResponse },
      { label: "Depth field expands inland", body: "The ensemble-mean depth pattern develops across a broader part of the modeled domain.", frameIndex: inlandExpansion },
      { label: "Peak mean depth", body: "The area-weighted mean water depth reaches its event maximum across the modeled floodplain.", frameIndex: peakDepth },
      { label: "Depth recession", body: "Mean water depth recedes across the retained forecast horizon.", frameIndex: recession }
    ];
  }, [peakMeanDepthTimeIndex, products]);
  const activeStep = milestoneIndexFromFrame(
    frameIndex,
    milestones.map((milestone) => milestone.frameIndex)
  );
  const activeMilestone = milestones[activeStep] ?? milestones[0];

  useEffect(() => {
    const node = rootRef.current;
    if (!node) return;
    if (!("IntersectionObserver" in window)) {
      setVisible(true);
      return;
    }
    const observer = new IntersectionObserver(([entry]) => setVisible(entry.isIntersecting), { threshold: 0.18 });
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    video.playbackRate = SPEEDS[speedIndex].rate;
    if (!playing || !visible) {
      video.pause();
      return;
    }
    void video.play().catch(() => setPlaying(false));
  }, [playing, productId, speedIndex, visible]);

  if (!product || !frame) return null;
  const seekToFrame = (nextFrameIndex: number) => {
    const boundedIndex = Math.min(product.frames.length - 1, Math.max(0, nextFrameIndex));
    setFrameIndex(boundedIndex);
    const video = videoRef.current;
    if (video) video.currentTime = boundedIndex / product.animation.sourceFrameRate;
  };
  const step = (delta: number) => {
    setPlaying(false);
    seekToFrame((frameIndex + delta + product.frames.length) % product.frames.length);
  };

  return (
    <div
      ref={rootRef}
      className="forecast-story-player"
      data-scroll-scene="portsmouth-forecast"
      data-playback-mode="autoplay-on-visible"
      data-playing={playing && visible}
      data-frame-index={frameIndex}
    >
      <div className="forecast-scroll-sticky">
        <div className="forecast-player-toolbar">
        <div className="case-study-segmented" aria-label="Forecast product">
          {products.map((item) => (
            <button
              type="button"
              key={item.id}
              className={item.id === product.id ? "active" : ""}
              aria-pressed={item.id === product.id}
              onClick={() => setProductId(item.id)}
            >
              {item.label}
            </button>
          ))}
        </div>
        <span className="forecast-lead" aria-live="polite">Lead +{frame.leadHours.toFixed(2)} h</span>
        </div>

        <figure className="case-study-figure forecast-player-figure">
          <div className="forecast-player-canvas">
          <video
            key={product.id}
            ref={videoRef}
            data-testid="forecast-product-video"
            muted
            loop
            playsInline
            preload="auto"
            poster={caseStudyAsset(product.animation.posterSrc || posterSrc)}
            aria-label={`${eventLabel} ${product.label} forecast animation`}
            width={1400}
            height={1080}
            onLoadedMetadata={(event) => {
              event.currentTarget.currentTime = frameIndex / product.animation.sourceFrameRate;
              event.currentTarget.playbackRate = SPEEDS[speedIndex].rate;
              if (playing && visible) void event.currentTarget.play().catch(() => setPlaying(false));
            }}
            onTimeUpdate={(event) => {
              const nextFrame = Math.min(
                product.frames.length - 1,
                Math.max(0, Math.round(event.currentTarget.currentTime * product.animation.sourceFrameRate))
              );
              setFrameIndex((current) => (current === nextFrame ? current : nextFrame));
            }}
          >
            <source src={caseStudyAsset(product.animation.mp4Src)} type="video/mp4" />
          </video>
            <div className="forecast-scroll-caption" aria-live="polite">
              <span>{String(activeStep + 1).padStart(2, "0")} / {String(milestones.length).padStart(2, "0")}</span>
              <strong>{activeMilestone.label}</strong>
              <p>{activeMilestone.body}</p>
            </div>
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
              seekToFrame(Number(event.target.value));
            }}
          />
          <button
            type="button"
            onClick={() => {
              seekToFrame(0);
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
            title="Follow the forecast from onset through recession"
            insight="Track where the selected product emerges, concentrates, and recedes across the full Irene 2011 horizon."
            method="Each source state is a 15-minute model lead. Adjacent rendered states are blended for smooth playback, while the timeline and controls remain anchored to the original forecast leads."
          />
        </figure>
      </div>
    </div>
  );
}
