"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Badge } from "@/components/ui/badge";
import { ImageWithRipple } from "@/components/image-with-ripple";

interface ImageData {
  path: string;
  cluster?: number;
  aesthetic_score?: number;
  is_duplicate?: boolean;
  is_low_aesthetic?: boolean;
}

interface ImageMasonryProps {
  images: ImageData[];
  onImageClick: (image: ImageData) => void;
}

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

function MasonryImage({
  image,
  index,
  onImageClick,
}: {
  image: ImageData;
  index: number;
  onImageClick: (image: ImageData) => void;
}) {
  const [isInView, setIsInView] = useState(false);
  const [isLoaded, setIsLoaded] = useState(false);
  const [isHovered, setIsHovered] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsInView(true);
          observer.disconnect();
        }
      },
      { rootMargin: "100px" }
    );

    if (ref.current) {
      observer.observe(ref.current);
    }

    return () => observer.disconnect();
  }, []);

  const formatAestheticScore = (score?: number) => {
    return score ? `${(score * 100).toFixed(0)}%` : "N/A";
  };

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: index * 0.02 }}
      className="masonry-item group cursor-pointer"
      onClick={() => onImageClick(image)}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <div className="relative overflow-hidden rounded-xl bg-muted/20 shadow-soft hover:shadow-strong transition-all duration-300 card-interactive">
        {isInView && (
          <ImageWithRipple
            src={`${BACKEND_URL}/api/image?path=${encodeURIComponent(image.path)}`}
            alt={image.path}
            onLoad={() => setIsLoaded(true)}
            className="w-full h-auto transition-transform duration-300 group-hover:scale-105"
          />
        )}

        {/* Overlay with badges */}
        <AnimatePresence>
          {isHovered && isLoaded && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="absolute inset-0 bg-gradient-to-t from-black/70 via-transparent to-transparent"
            >
              <div className="absolute top-3 right-3 flex flex-col gap-2">
                {image.cluster !== undefined && image.cluster !== -1 && (
                  <Badge variant="secondary" className="glass text-xs">
                    Cluster {image.cluster}
                  </Badge>
                )}
                {image.is_duplicate && (
                  <Badge variant="destructive" className="text-xs">
                    Duplicate
                  </Badge>
                )}
                {image.is_low_aesthetic && (
                  <Badge variant="outline" className="glass text-xs border-yellow-500/50 text-yellow-600">
                    Low Score
                  </Badge>
                )}
              </div>

              <div className="absolute bottom-3 left-3 right-3">
                <div className="text-white">
                  <p className="text-sm font-medium mb-1">
                    Score: {formatAestheticScore(image.aesthetic_score)}
                  </p>
                  <p className="text-xs opacity-75 truncate">
                    {image.path.split('/').pop()}
                  </p>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Loading placeholder */}
        {!isLoaded && isInView && (
          <div className="absolute inset-0 loading-skeleton" />
        )}
      </div>
    </motion.div>
  );
}

export function ImageMasonry({ images, onImageClick }: ImageMasonryProps) {
  const [columns, setColumns] = useState(4);

  const updateColumns = useCallback(() => {
    const width = window.innerWidth;
    if (width < 640) setColumns(1);
    else if (width < 768) setColumns(2);
    else if (width < 1024) setColumns(3);
    else if (width < 1280) setColumns(4);
    else setColumns(5);
  }, []);

  useEffect(() => {
    updateColumns();
    window.addEventListener('resize', updateColumns);
    return () => window.removeEventListener('resize', updateColumns);
  }, [updateColumns]);

  // Distribute images across columns
  const columnArrays = Array.from({ length: columns }, () => [] as ImageData[]);
  images.forEach((image, index) => {
    columnArrays[index % columns].push(image);
  });

  return (
    <div className="w-full">
      <div className="masonry-container" style={{ columns }}>
        {images.map((image, index) => (
          <MasonryImage
            key={image.path}
            image={image}
            index={index}
            onImageClick={onImageClick}
          />
        ))}
      </div>
    </div>
  );
}