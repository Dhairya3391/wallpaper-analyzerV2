"use client";

import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Masonry from "react-masonry-css";
import { LazyLoadImage } from "react-lazy-load-image-component";
import { Badge } from "@/components/ui/badge";
import { ImagePreview } from "@/components/image-preview";
import "react-lazy-load-image-component/src/effects/blur.css";

interface ImageData {
  path: string;
  cluster?: number;
  aesthetic_score?: number;
  is_duplicate?: boolean;
  is_low_aesthetic?: boolean;
}

interface ImageMasonryProps {
  images: ImageData[];
  isLoading: boolean;
}

const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

function MasonryImage({
  image,
  onImageClick,
}: {
  image: ImageData;
  onImageClick: (image: ImageData) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [isInView, setIsInView] = useState(false);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    const node = ref.current;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsInView(true);
          observer.disconnect();
        }
      },
      { rootMargin: "200px" },
    );

    if (node) {
      observer.observe(node);
    }

    return () => {
      if (node) {
        observer.unobserve(node);
      }
    };
  }, []);

  const formatAestheticScore = (score?: number) => {
    return score ? `Score: ${score.toFixed(2)}` : "N/A";
  };

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 20, scale: 0.98 }}
      animate={{
        opacity: isInView && loaded ? 1 : 0,
        y: isInView && loaded ? 0 : 20,
        scale: loaded ? 1 : 0.98,
      }}
      transition={{ duration: 0.5 }}
      className="break-inside-avoid relative group cursor-pointer"
      onClick={() => onImageClick(image)}
    >
      <div className="relative overflow-hidden bg-muted group">
        {isInView && (
          <LazyLoadImage
            src={`${BACKEND_URL}/api/image?path=${encodeURIComponent(image.path)}`}
            alt={image.path}
            className={`w-full h-auto transition-transform duration-300 group-hover:scale-110 group-hover:shadow-2xl ${!loaded ? "loading-shimmer" : ""}`}
            effect="blur"
            afterLoad={() => setLoaded(true)}
            placeholder={
              <div className="w-full h-[200px] bg-muted loading-shimmer" />
            }
            visibleByDefault={false}
          />
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
            {image.cluster !== undefined && image.cluster !== -1 && (
              <Badge variant="secondary">Cluster {image.cluster}</Badge>
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
            <div key={i} className="break-inside-avoid">
              <div className="w-full h-64 bg-muted rounded-lg animate-pulse" />
            </div>
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
        className="flex gap-2"
        columnClassName="masonry-column flex flex-col gap-2"
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
