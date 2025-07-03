"use client";

import { motion, AnimatePresence } from "framer-motion";
import Image from "next/image";
import { X, Download, Heart, Share } from "lucide-react";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { AspectRatio } from "@/components/ui/aspect-ratio";
import { useState } from "react";

interface ImageData {
  path: string;
  cluster?: number;
  aesthetic_score?: number;
  is_duplicate?: boolean;
  is_low_aesthetic?: boolean;
}

interface ImagePreviewProps {
  image: ImageData | null;
  onClose: () => void;
}

const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

export function ImagePreview({ image, onClose }: ImagePreviewProps) {
  const [isFavorite, setIsFavorite] = useState(false);

  const fileName = image?.path.split("/").pop() || "image.jpg";
  const imageUrl = image
    ? `${BACKEND_URL}/api/image?path=${encodeURIComponent(image.path)}`
    : "";

  const handleDownload = () => {
    const link = document.createElement("a");
    link.href = imageUrl;
    link.download = fileName;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleShare = async () => {
    const url = `${window.location.origin}/api/image?path=${encodeURIComponent(
      image?.path || "",
    )}`;
    if (navigator.share) {
      try {
        await navigator.share({
          title: fileName,
          url,
        });
      } catch {
        /* sharing cancelled or failed */
      }
    } else {
      try {
        await navigator.clipboard.writeText(url);
        alert("Image link copied to clipboard"); // Replace with toast if available
      } catch {
        alert("Failed to copy image link");
      }
    }
  };

  return (
    <Dialog open={!!image} onOpenChange={onClose}>
      <DialogContent className="max-w-2xl w-full p-0 bg-background border-none shadow-2xl">
        <AnimatePresence>
          {image && (
            <motion.div
              key={image.path}
              initial={{ opacity: 0, scale: 0.97 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.97 }}
              transition={{ duration: 0.25, ease: "easeInOut" }}
              className="relative flex flex-col items-center justify-center"
            >
              {/* Close Button */}
              <Button
                variant="ghost"
                size="icon"
                className="absolute top-3 right-3 z-10 bg-black/40 hover:bg-black/60 text-white rounded-full"
                onClick={onClose}
                aria-label="Close preview"
              >
                <X className="h-6 w-6" />
              </Button>

              {/* Image */}
              <div className="w-full max-w-2xl mx-auto mt-6">
                <AspectRatio
                  ratio={16 / 10}
                  className="bg-muted rounded-lg overflow-hidden"
                >
                  <Image
                    src={imageUrl}
                    alt={fileName}
                    fill
                    className="object-contain w-full h-full"
                    priority
                  />
                </AspectRatio>
              </div>

              {/* Filename & Badges */}
              <div className="flex flex-col items-center gap-2 mt-4">
                <span className="text-base font-semibold break-all text-center">
                  {fileName}
                </span>
                <div className="flex flex-wrap gap-2 justify-center">
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
              </div>

              {/* Actions */}
              <div className="flex gap-3 mt-6 mb-4">
                <Button
                  variant="outline"
                  onClick={handleDownload}
                  aria-label="Download image"
                >
                  <Download className="w-5 h-5 mr-2" />
                  Download
                </Button>
                <Button
                  variant={isFavorite ? "default" : "outline"}
                  onClick={() => setIsFavorite((f) => !f)}
                  aria-label="Favorite image"
                >
                  <Heart
                    className={`w-5 h-5 mr-2 ${
                      isFavorite ? "fill-red-500 text-red-500" : ""
                    }`}
                  />
                  {isFavorite ? "Favorited" : "Favorite"}
                </Button>
                <Button
                  variant="outline"
                  onClick={handleShare}
                  aria-label="Share image"
                >
                  <Share className="w-5 h-5 mr-2" />
                  Share
                </Button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </DialogContent>
    </Dialog>
  );
}
