"use client";

import { useEffect, useRef } from "react";

interface UseInfiniteScrollProps {
  target: React.RefObject<HTMLElement>;
  onIntersect: () => void;
  enabled?: boolean;
  threshold?: number;
  rootMargin?: string;
}

export function useInfiniteScroll({
  target,
  onIntersect,
  enabled = true,
  threshold = 0.1,
  rootMargin = "100px",
}: UseInfiniteScrollProps) {
  const observerRef = useRef<IntersectionObserver | null>(null);

  useEffect(() => {
    if (!enabled || !target.current) return;

    observerRef.current = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          onIntersect();
        }
      },
      {
        threshold,
        rootMargin,
      }
    );

    observerRef.current.observe(target.current);

    return () => {
      if (observerRef.current) {
        observerRef.current.disconnect();
      }
    };
  }, [target, onIntersect, enabled, threshold, rootMargin]);

  return observerRef.current;
}