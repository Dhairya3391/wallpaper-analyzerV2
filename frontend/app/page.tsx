"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Header } from "@/components/header";
import { ImageMasonry } from "@/components/image-masonry";
import { SearchBar } from "@/components/search-bar";
import { FilterTabs } from "@/components/filter-tabs";
import { SettingsPanel } from "@/components/settings-panel";
import { ImagePreview } from "@/components/image-preview";
import { ThemeProvider } from "@/components/theme-provider";
import { LoadingSpinner } from "@/components/loading-spinner";
import { EmptyState } from "@/components/empty-state";
import { useInfiniteScroll } from "@/hooks/use-infinite-scroll";
import { useDebounce } from "@/hooks/use-debounce";

const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

interface ImageData {
  path: string;
  cluster?: number;
  cluster_size?: number;
  is_duplicate?: boolean;
  is_low_aesthetic?: boolean;
  aesthetic_score?: number;
}

interface ClusterData {
  id: number;
  size: number;
}

export default function Home() {
  const [allImages, setAllImages] = useState<ImageData[]>([]);
  const [displayedImages, setDisplayedImages] = useState<ImageData[]>([]);
  const [clusters, setClusters] = useState<ClusterData[]>([]);
  const [selectedCluster, setSelectedCluster] = useState<string>("all");
  const [showDuplicates, setShowDuplicates] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [selectedImage, setSelectedImage] = useState<ImageData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [hasAnalyzed, setHasAnalyzed] = useState(false);

  const debouncedSearchTerm = useDebounce(searchTerm, 300);
  const loadMoreRef = useRef<HTMLDivElement>(null);

  const [settings, setSettings] = useState({
    directory: "/Users/dhairya/Downloads/walls",
    similarity_threshold: 0.85,
    aesthetic_threshold: 0.85,
    recursive: true,
    skip_duplicates: false,
    skip_aesthetics: false,
    limit: 1000,
  });

  // Filter images based on search and cluster selection
  const filteredImages = useCallback(() => {
    let filtered = allImages;

    // Apply search filter
    if (debouncedSearchTerm) {
      filtered = filtered.filter((image) =>
        image.path.toLowerCase().includes(debouncedSearchTerm.toLowerCase())
      );
    }

    // Apply cluster filter
    if (showDuplicates) {
      filtered = filtered.filter((img) => img.is_duplicate);
    } else if (selectedCluster !== "all") {
      filtered = filtered.filter(
        (img) => String(img.cluster) === selectedCluster
      );
    }

    return filtered;
  }, [allImages, debouncedSearchTerm, selectedCluster, showDuplicates]);

  // Infinite scroll implementation
  const IMAGES_PER_PAGE = 20;
  const [currentPage, setCurrentPage] = useState(1);

  const loadMoreImages = useCallback(() => {
    const filtered = filteredImages();
    const startIndex = (currentPage - 1) * IMAGES_PER_PAGE;
    const endIndex = startIndex + IMAGES_PER_PAGE;
    const newImages = filtered.slice(startIndex, endIndex);

    if (currentPage === 1) {
      setDisplayedImages(newImages);
    } else {
      setDisplayedImages((prev) => [...prev, ...newImages]);
    }
  }, [filteredImages, currentPage]);

  // Reset pagination when filters change
  useEffect(() => {
    setCurrentPage(1);
    setDisplayedImages([]);
  }, [debouncedSearchTerm, selectedCluster, showDuplicates]);

  // Load images when page changes
  useEffect(() => {
    loadMoreImages();
  }, [loadMoreImages]);

  // Infinite scroll hook
  const hasMore = displayedImages.length < filteredImages().length;

  useInfiniteScroll({
    target: loadMoreRef,
    onIntersect: () => {
      if (hasMore && !isLoading) {
        setCurrentPage((prev) => prev + 1);
      }
    },
    enabled: hasMore && !isLoading,
  });

  const analyzeDirectory = async () => {
    if (!settings.directory.startsWith("/")) {
      setError("Please enter an absolute directory path");
      return;
    }

    setIsLoading(true);
    setError(null);
    setHasAnalyzed(false);

    try {
      const response = await fetch(`${BACKEND_URL}/api/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(settings),
      });

      const data = await response.json();

      if (!data.success) {
        throw new Error(data.error || "Analysis failed");
      }

      setAllImages(data.images || []);
      setHasAnalyzed(true);

      // Extract clusters
      const clusterMap = new Map<number, number>();
      (data.images || []).forEach((img: ImageData) => {
        const clusterId =
          typeof img.cluster === "number" ? img.cluster : undefined;
        if (clusterId !== undefined && clusterId !== -1) {
          clusterMap.set(clusterId, (clusterMap.get(clusterId) || 0) + 1);
        }
      });

      const clustersArr = Array.from(clusterMap.entries()).map(
        ([id, size]) => ({
          id,
          size,
        })
      );

      setClusters(clustersArr);
      setCurrentPage(1);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
      setAllImages([]);
      setClusters([]);
    } finally {
      setIsLoading(false);
    }
  };

  const totalImages = allImages.length;
  const filteredCount = filteredImages().length;
  const hasDuplicates = allImages.some((img) => img.is_duplicate);

  return (
    <ThemeProvider attribute="class" defaultTheme="light" enableSystem>
      <div className="min-h-screen bg-background transition-colors duration-300">
        <Header onSettingsClick={() => setIsSettingsOpen(true)} />

        <main className="container mx-auto px-6 py-12">
          {/* Hero Section */}
          <div className="text-center mb-20">
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, ease: "easeOut" }}
              className="space-professional-lg"
            >
              <h1 className="text-5xl md:text-6xl lg:text-7xl font-bold text-professional-heading mb-8 leading-tight">
                Beautiful wallpapers,{" "}
                <span className="gradient-text">organized by AI</span>
              </h1>

              <motion.p
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2, duration: 0.6 }}
                className="text-xl md:text-2xl text-professional-muted mb-12 max-w-3xl mx-auto leading-relaxed"
              >
                Discover, analyze, and curate your perfect image collection with
                advanced AI algorithms and beautiful design.
              </motion.p>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4, duration: 0.6 }}
              >
                <SearchBar
                  value={settings.directory}
                  onChange={(value) =>
                    setSettings((s) => ({ ...s, directory: value }))
                  }
                  onAnalyze={analyzeDirectory}
                  isLoading={isLoading}
                  placeholder="Enter directory path (e.g. /Users/yourname/Pictures)"
                />
              </motion.div>

              {error && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="mt-8 p-4 glass border border-destructive/20 rounded-2xl text-destructive max-w-md mx-auto"
                >
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-destructive" />
                    {error}
                  </div>
                </motion.div>
              )}
            </motion.div>
          </div>

          {/* Filter Tabs */}
          {hasAnalyzed && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="mb-16"
            >
              <FilterTabs
                clusters={clusters}
                selectedCluster={selectedCluster}
                onClusterChange={setSelectedCluster}
                hasDuplicates={hasDuplicates}
                showDuplicates={showDuplicates}
                onShowDuplicates={setShowDuplicates}
                searchTerm={searchTerm}
                onSearchChange={setSearchTerm}
                totalImages={totalImages}
                filteredCount={filteredCount}
              />
            </motion.div>
          )}

          {/* Loading State */}
          {isLoading && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex flex-col items-center justify-center py-32"
            >
              <div className="relative">
                <LoadingSpinner size="large" />
                <motion.div
                  className="absolute inset-0 rounded-full border-2 border-primary/20"
                  animate={{ rotate: 360 }}
                  transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
                />
              </div>
              <motion.p
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 }}
                className="mt-8 text-professional-muted text-lg"
              >
                Analyzing your images with AI...
              </motion.p>
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.7 }}
                className="mt-2 text-sm text-professional-muted opacity-60"
              >
                This may take a few moments
              </motion.div>
            </motion.div>
          )}

          {/* Empty State */}
          {!isLoading && hasAnalyzed && displayedImages.length === 0 && (
            <EmptyState
              title="No images found"
              description="Try adjusting your filters or analyzing a different directory."
            />
          )}

          {/* Images Grid */}
          {!isLoading && displayedImages.length > 0 && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.6 }}
            >
              <ImageMasonry
                images={displayedImages}
                onImageClick={setSelectedImage}
              />

              {/* Load More Trigger */}
              {hasMore && (
                <div ref={loadMoreRef} className="flex justify-center py-16">
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex items-center gap-3 px-6 py-3 glass rounded-full border border-border/50"
                  >
                    <LoadingSpinner />
                    <span className="text-sm text-professional-muted">
                      Loading more images...
                    </span>
                  </motion.div>
                </div>
              )}

              {/* End Message */}
              {!hasMore && displayedImages.length > 0 && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="text-center py-16"
                >
                  <div className="inline-flex items-center gap-2 px-6 py-3 glass rounded-full border border-border/50">
                    <div className="w-2 h-2 rounded-full bg-accent" />
                    <span className="text-sm text-professional-muted">
                      You&apos;ve reached the end of the collection
                    </span>
                  </div>
                </motion.div>
              )}
            </motion.div>
          )}
        </main>

        {/* Settings Panel */}
        <SettingsPanel
          settings={settings}
          onSettingsChange={setSettings}
          open={isSettingsOpen}
          onOpenChange={setIsSettingsOpen}
        />

        {/* Image Preview */}
        <AnimatePresence>
          {selectedImage && (
            <ImagePreview
              image={selectedImage}
              onClose={() => setSelectedImage(null)}
            />
          )}
        </AnimatePresence>
      </div>
    </ThemeProvider>
  );
}
