"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useInView } from "react-intersection-observer";
import { LoadingSpinner } from "@/components/loading-spinner";
import { cn } from "@/lib/utils";

interface OptimizedImageProps {
  src: string;
  alt: string;
  className?: string;
  aspectRatio?: "square" | "auto" | "video" | "portrait" | "landscape";
  loadingStrategy?: "lazy" | "eager" | "progressive";
  placeholderType?: "skeleton" | "blur" | "color" | "gradient";
  onLoad?: () => void;
  onError?: () => void;
  onClick?: () => void;
  priority?: boolean;
  sizes?: string;
}

export function OptimizedImage({
  src,
  alt,
  className,
  aspectRatio = "auto",
  loadingStrategy = "lazy",
  placeholderType = "skeleton",
  onLoad,
  onError,
  onClick,
  priority = false,
  sizes = "100vw",
}: OptimizedImageProps) {
  const [isLoaded, setIsLoaded] = useState(false);
  const [hasError, setHasError] = useState(false);
  const [imageRef, inView] = useInView({
    threshold: 0.1,
    rootMargin: "50px",
    triggerOnce: true,
  });

  const shouldLoad = priority || loadingStrategy === "eager" || inView;

  const aspectRatioClasses = {
    square: "aspect-square",
    auto: "aspect-auto",
    video: "aspect-video",
    portrait: "aspect-[3/4]",
    landscape: "aspect-[4/3]",
  };

  const handleLoad = () => {
    setIsLoaded(true);
    onLoad?.();
  };

  const handleError = () => {
    setHasError(true);
    onError?.();
  };

  const renderPlaceholder = () => {
    switch (placeholderType) {
      case "skeleton":
        return (
          <div className="w-full h-full bg-gradient-to-br from-muted/50 to-muted/30 animate-pulse">
            <div className="w-full h-full bg-muted/20 animate-shimmer" />
          </div>
        );
      case "blur":
        return <div className="w-full h-full bg-muted/30 backdrop-blur-sm" />;
      case "gradient":
        return (
          <div className="w-full h-full bg-gradient-to-br from-primary/10 via-accent/10 to-muted/20" />
        );
      case "color":
      default:
        return <div className="w-full h-full bg-muted/20" />;
    }
  };

  if (hasError) {
    return (
      <div
        className={cn(
          "w-full h-full bg-muted flex items-center justify-center",
          aspectRatioClasses[aspectRatio],
          className
        )}
      >
        <div className="flex flex-col items-center justify-center gap-2 p-4 text-center">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="w-8 h-8 text-destructive"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M6 18L18 6M6 6l12 12"
            />
          </svg>
          <span className="text-sm font-medium text-destructive">
            Failed to load
          </span>
        </div>
      </div>
    );
  }

  return (
    <div
      className={cn(
        "relative overflow-hidden",
        aspectRatioClasses[aspectRatio],
        className
      )}
    >
      {/* Placeholder */}
      <AnimatePresence>
        {!isLoaded && (
          <motion.div
            initial={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="absolute inset-0"
          >
            {renderPlaceholder()}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Loading Overlay */}
      <AnimatePresence>
        {!isLoaded && shouldLoad && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 bg-background/50 flex items-center justify-center z-10"
          >
            <LoadingSpinner variant="dots" size="small" />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Image */}
      {shouldLoad && (
        <motion.img
          ref={imageRef}
          src={src}
          alt={alt}
          className={cn(
            "w-full h-full object-cover transition-all duration-300",
            isLoaded ? "opacity-100" : "opacity-0"
          )}
          loading={loadingStrategy === "lazy" ? "lazy" : "eager"}
          decoding="async"
          sizes={sizes}
          onLoad={handleLoad}
          onError={handleError}
          onClick={onClick}
          style={{
            imageRendering: "auto",
            imageOrientation: "from-image",
          }}
        />
      )}

      {/* Progressive Loading Placeholder */}
      {!shouldLoad && (
        <div className="w-full h-full">{renderPlaceholder()}</div>
      )}

      {/* Click Overlay */}
      {onClick && (
        <motion.div
          className="absolute inset-0 bg-black/0 hover:bg-black/20 transition-colors duration-300 cursor-pointer"
          onClick={onClick}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          <div className="absolute inset-0 flex items-center justify-center opacity-0 hover:opacity-100 transition-opacity duration-300">
            <div className="bg-white/20 backdrop-blur-sm rounded-full p-3">
              <svg
                className="w-6 h-6 text-white"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7"
                />
              </svg>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
}
