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

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

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
      setDisplayedImages(prev => [...prev, ...newImages]);
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
        setCurrentPage(prev => prev + 1);
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
        const clusterId = typeof img.cluster === "number" ? img.cluster : undefined;
        if (clusterId !== undefined && clusterId !== -1) {
          clusterMap.set(clusterId, (clusterMap.get(clusterId) || 0) + 1);
        }
      });

      const clustersArr = Array.from(clusterMap.entries()).map(([id, size]) => ({
        id,
        size,
      }));

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
      <div className="min-h-screen bg-white">
        <Header onSettingsClick={() => setIsSettingsOpen(true)} />

        <main className="container mx-auto px-4 py-8">
          {/* Hero Section */}
          <div className="text-center mb-12">
            <motion.h1
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="text-5xl font-bold text-gray-900 mb-4"
            >
              Beautiful wallpapers,{" "}
              <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                organized by AI
              </span>
            </motion.h1>
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="text-xl text-gray-600 mb-8 max-w-2xl mx-auto"
            >
              Discover, analyze, and curate your perfect image collection with
              advanced AI algorithms and beautiful design.
            </motion.p>

            <SearchBar
              value={settings.directory}
              onChange={(value) =>
                setSettings((s) => ({ ...s, directory: value }))
              }
              onAnalyze={analyzeDirectory}
              isLoading={isLoading}
              placeholder="Enter directory path (e.g. /Users/yourname/Pictures)"
            />

            {error && (
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700"
              >
                {error}
              </motion.div>
            )}
          </div>

          {/* Filter Tabs */}
          {hasAnalyzed && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="mb-8"
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
            <div className="flex flex-col items-center justify-center py-20">
              <LoadingSpinner size="large" />
              <p className="mt-4 text-gray-600">Analyzing your images...</p>
            </div>
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
              transition={{ duration: 0.5 }}
            >
              <ImageMasonry
                images={displayedImages}
                onImageClick={setSelectedImage}
              />

              {/* Load More Trigger */}
              {hasMore && (
                <div
                  ref={loadMoreRef}
                  className="flex justify-center py-8"
                >
                  <LoadingSpinner />
                </div>
              )}

              {/* End Message */}
              {!hasMore && displayedImages.length > 0 && (
                <div className="text-center py-8 text-gray-500">
                  You've reached the end of the collection
                </div>
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