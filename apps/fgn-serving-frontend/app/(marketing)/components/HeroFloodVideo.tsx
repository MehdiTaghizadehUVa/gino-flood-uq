"use client";

import Image from "next/image";
import { Pause, Play } from "lucide-react";
import { useEffect, useRef, useState } from "react";

const HERO_PLAYBACK_RATE = 0.65;

export function HeroFloodVideo({
  posterSrc,
  mp4Src,
  webmSrc
}: {
  posterSrc: string;
  mp4Src: string;
  webmSrc: string;
}) {
  const rootRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const manualPreferenceRef = useRef<"play" | "pause" | null>(null);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    const root = rootRef.current;
    const video = videoRef.current;
    if (!root || !video) return;

    let visible = true;
    const applyPlaybackRate = () => {
      video.defaultPlaybackRate = HERO_PLAYBACK_RATE;
      video.playbackRate = HERO_PLAYBACK_RATE;
    };

    const syncPlayback = () => {
      const playbackRequested = manualPreferenceRef.current !== "pause";
      const shouldPlay = visible && document.visibilityState === "visible" && playbackRequested;
      if (shouldPlay) {
        void video.play().catch(() => setPlaying(false));
      } else {
        video.pause();
      }
    };

    const observer = "IntersectionObserver" in window
      ? new IntersectionObserver(
          ([entry]) => {
            visible = entry.isIntersecting && entry.intersectionRatio >= 0.12;
            syncPlayback();
          },
          { threshold: [0, 0.12] }
        )
      : null;
    observer?.observe(root);
    document.addEventListener("visibilitychange", syncPlayback);
    video.addEventListener("loadedmetadata", applyPlaybackRate);
    applyPlaybackRate();
    syncPlayback();

    return () => {
      observer?.disconnect();
      document.removeEventListener("visibilitychange", syncPlayback);
      video.removeEventListener("loadedmetadata", applyPlaybackRate);
      video.pause();
    };
  }, []);

  const togglePlayback = () => {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) {
      manualPreferenceRef.current = "play";
      void video.play().catch(() => setPlaying(false));
    } else {
      manualPreferenceRef.current = "pause";
      video.pause();
    }
  };

  return (
    <>
      <div ref={rootRef} className="hero-media-shell" aria-hidden="true">
        <Image
          className="hero-media-poster"
          src={posterSrc}
          alt=""
          fill
          priority
          sizes="100vw"
        />
        <video
          ref={videoRef}
          className={`hero-media-video${playing ? " is-playing" : ""}`}
          autoPlay
          muted
          loop
          playsInline
          preload="auto"
          poster={posterSrc}
          tabIndex={-1}
          onPlaying={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onEnded={() => setPlaying(false)}
        >
          <source src={webmSrc} type="video/webm" />
          <source src={mp4Src} type="video/mp4" />
        </video>
      </div>
      <button
        type="button"
        className="hero-motion-toggle"
        aria-label={playing ? "Pause flood animation" : "Play flood animation"}
        aria-pressed={playing}
        title={playing ? "Pause flood animation" : "Play flood animation"}
        onClick={togglePlayback}
      >
        {playing ? <Pause size={15} aria-hidden="true" /> : <Play size={15} fill="currentColor" aria-hidden="true" />}
        <span>{playing ? "Pause animation" : "Play animation"}</span>
      </button>
    </>
  );
}
