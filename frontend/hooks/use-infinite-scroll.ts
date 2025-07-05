"use client";

import { useEffect, useRef, useCallback, useState } from "react";
import { useInView } from "react-intersection-observer";

interface UseInfiniteScrollProps {
  target: React.RefObject<HTMLElement>;
  onIntersect: () => void;
  enabled?: boolean;
  threshold?: number;
  rootMargin?: string;
  delay?: number;
  maxRetries?: number;
  retryDelay?: number;
}

export function useInfiniteScroll({
  target,
  onIntersect,
  enabled = true,
  threshold = 0.1,
  rootMargin = "100px",
  delay = 100,
  maxRetries = 3,
  retryDelay = 1000,
}: UseInfiniteScrollProps) {
  const [retryCount, setRetryCount] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);
  const observerRef = useRef<IntersectionObserver | null>(null);

  const { ref: intersectionRef, inView } = useInView({
    threshold,
    rootMargin,
    triggerOnce: false,
  });

  const handleIntersect = useCallback(async () => {
    if (!enabled || isLoading) return;

    setIsLoading(true);
    setRetryCount(0);

    try {
      // Add delay to prevent rapid firing
      await new Promise(resolve => {
        timeoutRef.current = setTimeout(resolve, delay);
      });

      await onIntersect();
    } catch (error) {
      console.error("Infinite scroll error:", error);

      // Retry logic
      if (retryCount < maxRetries) {
        setTimeout(() => {
          setRetryCount(prev => prev + 1);
          setIsLoading(false);
        }, retryDelay);
        return;
      }
    } finally {
      setIsLoading(false);
    }
  }, [
    enabled,
    isLoading,
    onIntersect,
    delay,
    retryCount,
    maxRetries,
    retryDelay,
  ]);

  // Handle intersection
  useEffect(() => {
    if (inView && enabled && !isLoading) {
      handleIntersect();
    }
  }, [inView, enabled, isLoading, handleIntersect]);

  // Set up intersection observer for the target element
  useEffect(() => {
    if (!enabled || !target.current) return;

    observerRef.current = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && !isLoading) {
          handleIntersect();
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
  }, [target, handleIntersect, enabled, threshold, rootMargin, isLoading]);

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  return {
    isLoading,
    retryCount,
    intersectionRef,
    inView,
  };
}
