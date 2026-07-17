"use client";

import Image from "next/image";
import { useEffect, useRef } from "react";

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
      const shouldPlay = visible && document.visibilityState === "visible";
      if (shouldPlay) {
        void video.play().catch(() => undefined);
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

  return (
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
        className="hero-media-video"
        autoPlay
        muted
        loop
        playsInline
        preload="auto"
        poster={posterSrc}
        tabIndex={-1}
      >
        <source src={webmSrc} type="video/webm" />
        <source src={mp4Src} type="video/mp4" />
      </video>
    </div>
  );
}
