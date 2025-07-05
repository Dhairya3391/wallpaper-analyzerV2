"use client";

import { motion, AnimatePresence } from "framer-motion";
import Image from "next/image";
import { X, Download, Heart, Share, Info } from "lucide-react";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
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

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

export function ImagePreview({ image, onClose }: ImagePreviewProps) {
  const [isFavorite, setIsFavorite] = useState(false);
  const [showInfo, setShowInfo] = useState(false);

  if (!image) return null;

  const fileName = image.path.split("/").pop() || "image.jpg";
  const imageUrl = `${BACKEND_URL}/api/image?path=${encodeURIComponent(image.path)}`;

  const handleDownload = () => {
    const link = document.createElement("a");
    link.href = imageUrl;
    link.download = fileName;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleShare = async () => {
    const url = `${window.location.origin}/api/image?path=${encodeURIComponent(image.path)}`;
    if (navigator.share) {
      try {
        await navigator.share({
          title: fileName,
          url,
        });
      } catch {
        // sharing cancelled or failed
      }
    } else {
      try {
        await navigator.clipboard.writeText(url);
        // You could add a toast notification here
      } catch {
        // Failed to copy
      }
    }
  };

  return (
    <Dialog open={!!image} onOpenChange={onClose}>
      <DialogContent className="max-w-7xl w-full h-[90vh] p-0 bg-black border-none">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.9 }}
          transition={{ duration: 0.3, ease: "easeOut" }}
          className="relative w-full h-full flex flex-col"
        >
          {/* Header */}
          <div className="absolute top-0 left-0 right-0 z-10 bg-gradient-to-b from-black/50 to-transparent p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <Button
                  variant="ghost"
                  size="icon"
                  className="text-white hover:bg-white/20 rounded-full"
                  onClick={onClose}
                >
                  <X className="h-6 w-6" />
                </Button>
                <div className="text-white">
                  <h3 className="font-semibold truncate max-w-md">{fileName}</h3>
                  <div className="flex items-center space-x-2 mt-1">
                    {image.cluster !== undefined && image.cluster !== -1 && (
                      <Badge variant="secondary" className="bg-white/20 text-white border-white/30">
                        Cluster {image.cluster}
                      </Badge>
                    )}
                    {image.is_duplicate && (
                      <Badge variant="destructive" className="bg-red-500/80">
                        Duplicate
                      </Badge>
                    )}
                    {image.is_low_aesthetic && (
                      <Badge variant="outline" className="bg-yellow-500/80 text-white border-yellow-500">
                        Low Score
                      </Badge>
                    )}
                  </div>
                </div>
              </div>

              <div className="flex items-center space-x-2">
                <Button
                  variant="ghost"
                  size="icon"
                  className="text-white hover:bg-white/20 rounded-full"
                  onClick={() => setShowInfo(!showInfo)}
                >
                  <Info className="h-5 w-5" />
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  className="text-white hover:bg-white/20 rounded-full"
                  onClick={handleShare}
                >
                  <Share className="h-5 w-5" />
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  className={`rounded-full ${
                    isFavorite
                      ? "text-red-500 hover:bg-red-500/20"
                      : "text-white hover:bg-white/20"
                  }`}
                  onClick={() => setIsFavorite(!isFavorite)}
                >
                  <Heart className={`h-5 w-5 ${isFavorite ? "fill-current" : ""}`} />
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  className="text-white hover:bg-white/20 rounded-full"
                  onClick={handleDownload}
                >
                  <Download className="h-5 w-5" />
                </Button>
              </div>
            </div>
          </div>

          {/* Image */}
          <div className="flex-1 flex items-center justify-center p-4">
            <div className="relative max-w-full max-h-full">
              <Image
                src={imageUrl}
                alt={fileName}
                width={1200}
                height={800}
                className="max-w-full max-h-full object-contain"
                priority
              />
            </div>
          </div>

          {/* Info Panel */}
          <AnimatePresence>
            {showInfo && (
              <motion.div
                initial={{ opacity: 0, x: 300 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 300 }}
                transition={{ duration: 0.3 }}
                className="absolute top-0 right-0 w-80 h-full bg-black/90 backdrop-blur-sm p-6 overflow-y-auto"
              >
                <div className="text-white space-y-4">
                  <h4 className="text-lg font-semibold">Image Details</h4>
                  
                  <div className="space-y-3">
                    <div>
                      <label className="text-sm text-gray-300">Filename</label>
                      <p className="text-sm break-all">{fileName}</p>
                    </div>
                    
                    <div>
                      <label className="text-sm text-gray-300">Path</label>
                      <p className="text-sm break-all">{image.path}</p>
                    </div>
                    
                    {image.aesthetic_score !== undefined && (
                      <div>
                        <label className="text-sm text-gray-300">Aesthetic Score</label>
                        <p className="text-sm">{(image.aesthetic_score * 100).toFixed(1)}%</p>
                      </div>
                    )}
                    
                    {image.cluster !== undefined && image.cluster !== -1 && (
                      <div>
                        <label className="text-sm text-gray-300">Cluster</label>
                        <p className="text-sm">Cluster {image.cluster}</p>
                      </div>
                    )}
                    
                    <div>
                      <label className="text-sm text-gray-300">Status</label>
                      <div className="flex flex-wrap gap-1 mt-1">
                        {image.is_duplicate && (
                          <Badge variant="destructive" className="text-xs">
                            Duplicate
                          </Badge>
                        )}
                        {image.is_low_aesthetic && (
                          <Badge variant="outline" className="text-xs">
                            Low Aesthetic
                          </Badge>
                        )}
                        {!image.is_duplicate && !image.is_low_aesthetic && (
                          <Badge variant="secondary" className="text-xs">
                            Original
                          </Badge>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      </DialogContent>
    </Dialog>
  );
}