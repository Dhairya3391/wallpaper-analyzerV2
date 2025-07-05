"use client";

import React, { useState, useCallback, useMemo, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { LazyLoadImage } from "react-lazy-load-image-component";
import { Masonry } from "masonic";
import Lightbox from "yet-another-react-lightbox";
import Zoom from "yet-another-react-lightbox/plugins/zoom";
import Thumbnails from "yet-another-react-lightbox/plugins/thumbnails";
import Fullscreen from "yet-another-react-lightbox/plugins/fullscreen";
import "yet-another-react-lightbox/styles.css";
import "yet-another-react-lightbox/plugins/thumbnails.css";
import { Card } from "@/components/ui/card";

import { cn } from "@/lib/utils";

export interface MasonryGalleryProps {
  images: string[];
  className?: string;
  aspectRatio?: "square" | "auto" | "video";
  showThumbnails?: boolean;
  enableZoom?: boolean;
  enableFullscreen?: boolean;
  columnGutter?: number;
  columnWidth?: number;
  overscanBy?: number;
}

interface ImageItem {
  id: string;
  src: string;
  alt: string;
  index: number;
}

const MasonryGallery: React.FC<MasonryGalleryProps> = ({
  images,
  className,
  aspectRatio = "auto",
  showThumbnails = true,
  enableZoom = true,
  enableFullscreen = true,
  columnGutter = 16,
  columnWidth = 300,
  overscanBy = 2,
}) => {
  const [lightboxOpen, setLightboxOpen] = useState(false);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);

  const items = useMemo(
    () =>
      images
        .filter((src) => src && typeof src === "string" && src.trim() !== "")
        .map((src, index) => ({
          id: `image-${index}`,
          src,
          alt: `Gallery image ${index + 1}`,
          index,
        })),
    [images],
  );

  // Lightbox images
  const lightboxImages = useMemo(
    () =>
      images
        .filter((src) => src && typeof src === "string" && src.trim() !== "")
        .map((src, index) => ({ src, alt: `Image ${index + 1}` })),
    [images],
  );

  const handleImageClick = useCallback((index: number) => {
    setCurrentImageIndex(index);
    setLightboxOpen(true);
  }, []);

  const plugins = useMemo(() => {
    const pluginList = [];
    if (enableZoom) pluginList.push(Zoom);
    if (showThumbnails) pluginList.push(Thumbnails);
    if (enableFullscreen) pluginList.push(Fullscreen);
    return pluginList;
  }, [enableZoom, showThumbnails, enableFullscreen]);

  return (
    <div className={cn("w-full", className)}>
      {items.length > 0 && (
        <Masonry
          items={items}
          columnWidth={columnWidth}
          columnGutter={columnGutter}
          overscanBy={overscanBy}
          render={({ data }: { data: ImageItem }) => (
            <MasonryImageItem
              key={data.id}
              src={data.src}
              alt={data.alt}
              index={data.index}
              aspectRatio={aspectRatio}
              onClick={() => handleImageClick(data.index)}
            />
          )}
        />
      )}
      <Lightbox
        open={lightboxOpen}
        close={() => setLightboxOpen(false)}
        index={currentImageIndex}
        slides={lightboxImages}
        plugins={plugins}
        carousel={{ finite: true }}
        zoom={{
          maxZoomPixelRatio: 3,
          zoomInMultiplier: 2,
          doubleTapDelay: 300,
          doubleClickDelay: 300,
          doubleClickMaxStops: 2,
          keyboardMoveDistance: 50,
          wheelZoomDistanceFactor: 100,
          pinchZoomDistanceFactor: 100,
          scrollToZoom: true,
        }}
        thumbnails={{
          width: 120,
          height: 80,
          padding: 4,
          border: 2,
          borderRadius: 8,
          gap: 16,
          imageFit: "contain",
        }}
        animation={{ fade: 300, swipe: 500 }}
        controller={{ closeOnBackdropClick: true, closeOnPullDown: true }}
      />
    </div>
  );
};

interface MasonryImageItemProps {
  src: string;
  alt: string;
  index: number;
  aspectRatio: "square" | "auto" | "video";
  onClick: () => void;
}

const MasonryImageItem: React.FC<MasonryImageItemProps> = ({
  src,
  alt,
  index,
  aspectRatio,
  onClick,
}) => {
  const [isLoaded, setIsLoaded] = useState(false);
  const [hasError, setHasError] = useState(false);
  const [imageHeight, setImageHeight] = useState<number | null>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  // Timeout fallback: mark error if the image hasn’t loaded within 8 s (stable across re-renders)
  useEffect(() => {
    // Only set the timer once – on mount
    timerRef.current = setTimeout(() => {
      setHasError(true);
      setIsLoaded(true);
    }, 8000);
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  // Safety check for invalid src
  if (!src || typeof src !== "string" || src.trim() === "") {
    return null;
  }

  const aspectRatioClasses = {
    square: "aspect-square",
    auto: "aspect-auto",
    video: "aspect-video",
  };

  const handleLoad = (event: React.SyntheticEvent<HTMLImageElement>) => {
    if (timerRef.current) clearTimeout(timerRef.current);
    setIsLoaded(true);
    if (aspectRatio === "auto") {
      const img = event.currentTarget;
      const ratio = img.naturalHeight / img.naturalWidth;
      setImageHeight(img.offsetWidth * ratio);
    }
  };

  const handleError = () => {
    if (timerRef.current) clearTimeout(timerRef.current);
    setHasError(true);
    setIsLoaded(true);
  };

  const containerStyle =
    aspectRatio === "auto" && imageHeight ? { height: imageHeight } : undefined;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{
        duration: 0.3,
        delay: Math.min(index * 0.02, 0.5), // Cap the delay to prevent long animations
        ease: "easeOut",
      }}
      className="relative group mb-4"
      style={containerStyle}
    >
      <Card className="overflow-hidden cursor-pointer border-0 shadow-sm hover:shadow-lg transition-all duration-300 h-full">
        <motion.div
          whileHover={{ scale: 1.01 }}
          className="relative h-full"
          onClick={onClick}
        >
          {/* Loading Skeleton */}
          {!isLoaded && (
            <div
              className={cn("w-full bg-muted/30", aspectRatioClasses[aspectRatio])}
            >
              {/* static placeholder without pulse */}
            </div>
          )}

          {/* Error State */}
          {hasError && (
            <div
              className={cn(
                "w-full bg-muted flex items-center justify-center",
                aspectRatioClasses[aspectRatio],
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
                <span className="text-sm font-medium text-destructive">Failed to load</span>
              </div>
            </div>
          )}

          {/* Image */}
          {!hasError && (
            <div
              className={cn(
                "relative overflow-hidden",
                aspectRatioClasses[aspectRatio],
              )}
            >
              <LazyLoadImage
                src={src}
                alt={alt}
                effect="blur"
                className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-[1.02]"
                onLoad={handleLoad}
                onError={handleError}
                placeholder={
                  <div className="w-full h-full bg-muted/20" />
                }
                threshold={100}
                wrapperClassName="w-full h-full"
              />

              {/* Overlay */}
              <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-colors duration-300" />

              {/* Click indicator */}
              <motion.div
                className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-300"
                whileHover={{ scale: 1.05 }}
              >
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
              </motion.div>
            </div>
          )}
        </motion.div>
      </Card>
    </motion.div>
  );
};

export default MasonryGallery;
