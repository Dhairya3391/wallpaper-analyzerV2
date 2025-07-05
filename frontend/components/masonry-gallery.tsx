"use client";

import React, {
  useState,
  useCallback,
  useMemo,
  useEffect,
  useRef,
} from "react";
import { motion } from "framer-motion";
import { useInView } from "react-intersection-observer";
import { useSpring, animated } from "@react-spring/web";
import Image from "next/image";
import Lightbox from "yet-another-react-lightbox";
import Zoom from "yet-another-react-lightbox/plugins/zoom";
import Thumbnails from "yet-another-react-lightbox/plugins/thumbnails";
import Fullscreen from "yet-another-react-lightbox/plugins/fullscreen";
import "yet-another-react-lightbox/styles.css";
import "yet-another-react-lightbox/plugins/thumbnails.css";
import { Card } from "@/components/ui/card";
import { LoadingSpinner } from "@/components/loading-spinner";
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
  loadingStrategy?: "lazy" | "eager" | "progressive";
  placeholderType?: "skeleton" | "blur" | "color";
}

interface VirtualizedItem {
  id: string;
  src: string;
  alt: string;
  index: number;
  top: number;
  height: number;
  width: number;
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
  loadingStrategy = "lazy",
  placeholderType = "skeleton",
}) => {
  const [lightboxOpen, setLightboxOpen] = useState(false);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [loadedImages, setLoadedImages] = useState<Set<string>>(new Set());
  const [imageDimensions, setImageDimensions] = useState<
    Map<string, { width: number; height: number }>
  >(new Map());
  const [columns, setColumns] = useState<VirtualizedItem[][]>([]);
  const containerRef = useRef<HTMLDivElement>(null);

  const items = useMemo(
    () =>
      images
        .filter(src => src && typeof src === "string" && src.trim() !== "")
        .map((src, index) => ({
          id: `image-${index}`,
          src,
          alt: `Gallery image ${index + 1}`,
          index,
        })),
    [images]
  );

  // Lightbox images
  const lightboxImages = useMemo(
    () =>
      images
        .filter(src => src && typeof src === "string" && src.trim() !== "")
        .map((src, index) => ({ src, alt: `Image ${index + 1}` })),
    [images]
  );

  // Calculate masonry layout
  useEffect(() => {
    if (!containerRef.current || items.length === 0) return;

    const containerWidth = containerRef.current.offsetWidth;
    const numColumns = Math.floor(
      containerWidth / (columnWidth + columnGutter)
    );
    const actualColumnWidth =
      (containerWidth - (numColumns - 1) * columnGutter) / numColumns;

    const columnHeights = new Array(numColumns).fill(0);
    const newColumns: VirtualizedItem[][] = Array.from(
      { length: numColumns },
      () => []
    );

    items.forEach(item => {
      // Find shortest column
      const shortestColumnIndex = columnHeights.indexOf(
        Math.min(...columnHeights)
      );

      const dimensions = imageDimensions.get(item.id);
      const height = dimensions
        ? (dimensions.height / dimensions.width) * actualColumnWidth
        : 200; // Default height

      const virtualizedItem: VirtualizedItem = {
        ...item,
        top: columnHeights[shortestColumnIndex],
        height,
        width: actualColumnWidth,
      };

      newColumns[shortestColumnIndex].push(virtualizedItem);
      columnHeights[shortestColumnIndex] += height + columnGutter;
    });

    setColumns(newColumns);
  }, [items, imageDimensions, columnWidth, columnGutter]);

  const handleImageClick = useCallback((index: number) => {
    setCurrentImageIndex(index);
    setLightboxOpen(true);
  }, []);

  const handleImageLoad = useCallback(
    (id: string, width: number, height: number) => {
      setLoadedImages(prev => new Set(prev).add(id));
      setImageDimensions(prev => new Map(prev).set(id, { width, height }));
    },
    []
  );

  const plugins = useMemo(() => {
    const pluginList = [];
    if (enableZoom) pluginList.push(Zoom);
    if (showThumbnails) pluginList.push(Thumbnails);
    if (enableFullscreen) pluginList.push(Fullscreen);
    return pluginList;
  }, [enableZoom, showThumbnails, enableFullscreen]);

  return (
    <div className={cn("w-full", className)}>
      <div ref={containerRef} className="relative">
        <div className="flex gap-4">
          {columns.map((column, columnIndex) => (
            <div key={columnIndex} className="flex-1">
              {column.map(item => (
                <MasonryImageItem
                  key={item.id}
                  item={item}
                  aspectRatio={aspectRatio}
                  onClick={() => handleImageClick(item.index)}
                  onLoad={handleImageLoad}
                  isLoaded={loadedImages.has(item.id)}
                  loadingStrategy={loadingStrategy}
                  placeholderType={placeholderType}
                />
              ))}
            </div>
          ))}
        </div>
      </div>

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
  item: VirtualizedItem;
  aspectRatio: "square" | "auto" | "video";
  onClick: () => void;
  onLoad: (id: string, width: number, height: number) => void;
  isLoaded: boolean;
  loadingStrategy: "lazy" | "eager" | "progressive";
  placeholderType: "skeleton" | "blur" | "color";
}

const MasonryImageItem: React.FC<MasonryImageItemProps> = ({
  item,
  aspectRatio,
  onClick,
  onLoad,
  isLoaded,
  loadingStrategy,
  placeholderType,
}) => {
  const [isLoading, setIsLoading] = useState(true);
  const [hasError, setHasError] = useState(false);
  const [inView] = useInView({
    threshold: 0.1,
    rootMargin: "50px",
    triggerOnce: true,
  });

  const shouldLoad = loadingStrategy === "eager" || inView;

  const { opacity, scale } = useSpring({
    opacity: isLoaded ? 1 : 0,
    scale: isLoaded ? 1 : 0.95,
    config: { tension: 300, friction: 30 },
  });

  const aspectRatioClasses = {
    square: "aspect-square",
    auto: "aspect-auto",
    video: "aspect-video",
  };

  const handleLoad = (event: React.SyntheticEvent<HTMLImageElement>) => {
    setIsLoading(false);
    const img = event.currentTarget;
    onLoad(item.id, img.naturalWidth, img.naturalHeight);
  };

  const handleError = () => {
    setIsLoading(false);
    setHasError(true);
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
      case "color":
      default:
        return <div className="w-full h-full bg-muted/20" />;
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{
        duration: 0.3,
        delay: Math.min(item.index * 0.02, 0.5),
        ease: "easeOut",
      }}
      className="relative group mb-4"
      style={{ height: item.height }}
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
              className={cn("w-full h-full", aspectRatioClasses[aspectRatio])}
            >
              {renderPlaceholder()}
            </div>
          )}

          {/* Error State */}
          {hasError && (
            <div
              className={cn(
                "w-full h-full bg-muted flex items-center justify-center",
                aspectRatioClasses[aspectRatio]
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
          )}

          {/* Image */}
          {!hasError && shouldLoad && (
            <animated.div
              style={{ opacity, transform: scale.to(s => `scale(${s})`) }}
              className={cn(
                "relative overflow-hidden w-full h-full",
                aspectRatioClasses[aspectRatio]
              )}
            >
              <Image
                src={item.src}
                alt={item.alt}
                width={item.width}
                height={item.height}
                className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-[1.02]"
                onLoad={handleLoad}
                onError={handleError}
                loading={loadingStrategy === "lazy" ? "lazy" : "eager"}
                decoding="async"
              />

              {/* Loading Overlay */}
              {isLoading && (
                <div className="absolute inset-0 bg-background/50 flex items-center justify-center">
                  <LoadingSpinner variant="dots" size="small" />
                </div>
              )}

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
            </animated.div>
          )}

          {/* Progressive Loading Placeholder */}
          {!hasError && !shouldLoad && (
            <div
              className={cn("w-full h-full", aspectRatioClasses[aspectRatio])}
            >
              {renderPlaceholder()}
            </div>
          )}
        </motion.div>
      </Card>
    </motion.div>
  );
};

export default MasonryGallery;
