"use client";

import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Masonry from "react-masonry-css";
import { LazyLoadImage } from "react-lazy-load-image-component";
import { Badge } from "@/components/ui/badge";
import { ImagePreview } from "@/components/image-preview";
import "react-lazy-load-image-component/src/effects/blur.css";
import { Skeleton } from "@/components/ui/skeleton";

interface ImageData {
  path: string;
  cluster?: number;
  aesthetic_score?: number;
  is_duplicate?: boolean;
  is_low_aesthetic?: boolean;
  brightness?: number;
  label?: string;
}

interface ImageMasonryProps {
  images: ImageData[];
  isLoading: boolean;
}

const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

function formatBrightness(brightness?: number | null) {
  if (brightness === undefined || brightness === null || isNaN(brightness))
    return "N/A";
  return `Brightness: ${brightness.toFixed(1)}`;
}

function formatAestheticScore(score?: number) {
  return score ? `Score: ${score.toFixed(2)}` : "N/A";
}

function displayClusterLabel(image: ImageData) {
  if (image.cluster === undefined || image.cluster === -1) return null;
  const isLabeled =
    image.label && !["unknown", "Unlabeled"].includes(image.label);
  return isLabeled
    ? `${image.label} (Cluster ${image.cluster})`
    : `Cluster ${image.cluster}`;
}

function MasonryImage({
  image,
  onImageClick,
}: {
  image: ImageData;
  onImageClick: (image: ImageData) => void;
}) {
  const observerRef = useRef<HTMLDivElement>(null);
  const [isInView, setIsInView] = useState(false);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    const node = observerRef.current;
    if (!node) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsInView(true);
          observer.disconnect();
        }
      },
      { rootMargin: "200px" }
    );

    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  return (
    <motion.div
      ref={observerRef}
      initial={{ opacity: 0, y: 20, scale: 0.98 }}
      animate={{
        opacity: isInView && loaded ? 1 : 0,
        y: isInView && loaded ? 0 : 20,
        scale: loaded ? 1 : 0.98,
      }}
      transition={{ duration: 0.7, ease: "easeOut" }}
      className="break-inside-avoid relative group cursor-pointer"
      whileHover={{ scale: 1.03 }}
      onClick={() => onImageClick(image)}
      layout
    >
      <div className="relative overflow-hidden bg-muted group min-h-[200px]">
        {isInView ? (
          <>
            <LazyLoadImage
              src={`${BACKEND_URL}/api/image?path=${encodeURIComponent(
                image.path
              )}`}
              alt={`Image - ${image.label ?? "Unnamed"}`}
              className={`w-full h-auto transition-transform duration-500 group-hover:scale-110 group-hover:shadow-2xl ${
                !loaded ? "opacity-0" : "opacity-100"
              }`}
              effect="blur"
              afterLoad={() => setLoaded(true)}
              visibleByDefault={false}
            />
            {!loaded && (
              <Skeleton className="absolute inset-0 w-full h-full z-10" />
            )}
          </>
        ) : (
          <Skeleton className="w-full h-[200px]" />
        )}

        <motion.div
          initial={{ opacity: 0 }}
          whileHover={{ opacity: 1 }}
          animate={{ opacity: 0 }}
          whileTap={{ opacity: 1 }}
          transition={{ duration: 0.25 }}
          className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent flex flex-col justify-end p-3 pointer-events-none group-hover:pointer-events-auto"
        >
          <div className="flex flex-wrap gap-2 mb-1">
            {displayClusterLabel(image) && (
              <Badge variant="secondary">{displayClusterLabel(image)}</Badge>
            )}
            {image.is_duplicate && (
              <Badge variant="destructive">Duplicate</Badge>
            )}
            {image.is_low_aesthetic && (
              <Badge variant="outline">Low Score</Badge>
            )}
          </div>
          <p className="text-white text-base font-semibold drop-shadow-md">
            {formatAestheticScore(image.aesthetic_score)}
          </p>
          <p className="text-white text-xs drop-shadow-md mt-1">
            {formatBrightness(image.brightness)}
          </p>
        </motion.div>
      </div>
    </motion.div>
  );
}

export function ImageMasonry({ images, isLoading }: ImageMasonryProps) {
  const [selectedImage, setSelectedImage] = useState<ImageData | null>(null);

  const breakpointColumnsObj = {
    default: 3,
    1536: 3,
    1280: 2,
    900: 1,
  };

  if (isLoading) {
    return (
      <div className="mt-8">
        <Masonry
          breakpointCols={breakpointColumnsObj}
          className="flex gap-4"
          columnClassName="masonry-column flex flex-col gap-4"
        >
          {Array.from({ length: 12 }).map((_, i) => (
            <Skeleton key={i} className="w-full h-64 rounded-lg" />
          ))}
        </Masonry>
      </div>
    );
  }

  if (images.length === 0) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="text-center py-12"
      >
        <div className="text-muted-foreground">
          No images to display. Start by analyzing a directory.
        </div>
      </motion.div>
    );
  }

  return (
    <div className="mt-4">
      <Masonry
        breakpointCols={breakpointColumnsObj}
        className="flex gap-4"
        columnClassName="masonry-column flex flex-col gap-4"
      >
        {images.map((image) => (
          <MasonryImage
            key={image.path}
            image={image}
            onImageClick={setSelectedImage}
          />
        ))}
      </Masonry>

      <AnimatePresence>
        {selectedImage && (
          <ImagePreview
            image={selectedImage}
            onClose={() => setSelectedImage(null)}
          />
        )}
      </AnimatePresence>
    </div>
  );
}
