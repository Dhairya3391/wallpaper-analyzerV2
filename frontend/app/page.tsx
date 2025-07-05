"use client";

import {
  useState,
  useEffect,
  useCallback,
  useMemo,
  useRef as _useRef,
} from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Header } from "@/components/header";
import MasonryGallery from "@/components/masonry-gallery";
import { SearchBar } from "@/components/search-bar";
import { FilterTabs } from "@/components/filter-tabs";
import { SettingsPanel } from "@/components/settings-panel";
import { ThemeProvider } from "@/components/theme-provider";
import { LoadingSpinner } from "@/components/loading-spinner";
import { EmptyState } from "@/components/empty-state";
import { useToast } from "@/hooks/use-toast";
import { Toaster } from "@/components/ui/toaster";
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
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [hasAnalyzed, setHasAnalyzed] = useState(false);
  const [progress, setProgress] = useState(0);
  const [loadingPhase, setLoadingPhase] = useState<string>("");

  const { toast } = useToast();
  const debouncedSearchTerm = useDebounce(searchTerm, 300);

  const [settings, setSettings] = useState({
    directory: "/Users/dhairya/Downloads/walls",
    similarity_threshold: 0.85,
    aesthetic_threshold: 0.85,
    recursive: true,
    skip_duplicates: false,
    skip_aesthetics: false,
    limit: 1000,
  });

  const IMAGES_PER_PAGE = 20;
  const [currentPage, setCurrentPage] = useState(1);

  const filteredImages = useMemo(() => {
    let filtered = allImages.filter(
      img => img && typeof img === "object" && img.path
    );

    if (debouncedSearchTerm) {
      filtered = filtered.filter(image =>
        image.path.toLowerCase().includes(debouncedSearchTerm.toLowerCase())
      );
    }

    if (showDuplicates) {
      filtered = filtered.filter(img => img.is_duplicate);
    } else if (selectedCluster !== "all") {
      const clusterId = parseInt(selectedCluster, 10);
      filtered = filtered.filter(img => img.cluster === clusterId);
    }

    return filtered;
  }, [allImages, debouncedSearchTerm, selectedCluster, showDuplicates]);

  useEffect(() => {
    const newImages = filteredImages.slice(0, IMAGES_PER_PAGE);
    setDisplayedImages(newImages);
    setCurrentPage(1);
  }, [filteredImages]);

  const loadMoreImages = useCallback(() => {
    const nextPage = currentPage + 1;
    const startIndex = nextPage * IMAGES_PER_PAGE - IMAGES_PER_PAGE;
    const endIndex = nextPage * IMAGES_PER_PAGE;
    const newImages = filteredImages.slice(startIndex, endIndex);

    setDisplayedImages(prev => [...prev, ...newImages]);
    setCurrentPage(nextPage);
  }, [currentPage, filteredImages]);

  const hasMore = displayedImages.length < filteredImages.length;

  const dummyRef = _useRef<HTMLDivElement>(null);
  const { isLoading: isLoadingMore, intersectionRef } = useInfiniteScroll({
    target: dummyRef,
    onIntersect: loadMoreImages,
    enabled: hasMore && !isLoading,
    delay: 200,
  });

  const analyzeDirectory = async () => {
    if (!settings.directory.startsWith("/")) {
      setError("Please enter an absolute directory path");
      toast({
        variant: "destructive",
        title: "Invalid Path",
        description: "Please enter an absolute directory path starting with /",
      });
      return;
    }

    setIsLoading(true);
    setError(null);
    setHasAnalyzed(false);
    setProgress(0);
    setLoadingPhase("Initializing analysis...");

    toast({
      title: "Analysis Started",
      description: "Beginning to analyze your image directory...",
    });

    try {
      // Simulate progress updates
      const progressInterval = setInterval(() => {
        setProgress(prev => {
          if (prev >= 90) return prev;
          return prev + Math.random() * 10;
        });
      }, 500);

      const phaseInterval = setInterval(() => {
        setLoadingPhase(() => {
          const phases = [
            "Scanning directory...",
            "Loading images...",
            "Analyzing duplicates...",
            "Calculating aesthetics...",
            "Organizing clusters...",
            "Finalizing results...",
          ];
          const currentIndex = Math.floor((progress / 100) * phases.length);
          return phases[Math.min(currentIndex, phases.length - 1)];
        });
      }, 1000);

      const response = await fetch(`${BACKEND_URL}/api/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(settings),
      });

      clearInterval(progressInterval);
      clearInterval(phaseInterval);

      const data = await response.json();

      if (!data.success) {
        throw new Error(data.error || "Analysis failed");
      }

      setProgress(100);
      setLoadingPhase("Analysis complete!");

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

      // Show success toast
      toast({
        title: "Analysis Complete!",
        description: `Found ${
          data.images?.length || 0
        } images in your directory`,
      });

      // Reset progress after a delay
      setTimeout(() => {
        setProgress(0);
        setLoadingPhase("");
      }, 2000);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Unknown error";
      setError(errorMessage);
      setAllImages([]);
      setClusters([]);
      setProgress(0);
      setLoadingPhase("");

      toast({
        variant: "destructive",
        title: "Analysis Failed",
        description: errorMessage,
      });
    } finally {
      setIsLoading(false);
    }
  };

  const totalImages = allImages.length;
  const filteredCount = filteredImages.length;
  const hasDuplicates = allImages.some(img => img.is_duplicate);

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
              className="space-y-8"
            >
              <h1 className="text-5xl md:text-6xl lg:text-7xl font-bold text-foreground mb-8 leading-tight">
                Beautiful wallpapers,{" "}
                <span className="gradient-text">organized by AI</span>
              </h1>

              <motion.p
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2, duration: 0.6 }}
                className="text-xl md:text-2xl text-muted-foreground mb-12 max-w-3xl mx-auto leading-relaxed"
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
                  onChange={value =>
                    setSettings(s => ({ ...s, directory: value }))
                  }
                  onAnalyze={analyzeDirectory}
                  isLoading={isLoading}
                  placeholder="Enter directory path (e.g. /Users/yourname/Pictures)"
                  autoFocus={true}
                />
              </motion.div>

              <AnimatePresence>
                {error && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.95 }}
                    className="mt-8 p-4 glass border border-destructive/20 rounded-2xl text-destructive max-w-md mx-auto"
                  >
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full bg-destructive animate-pulse" />
                      {error}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          </div>

          {/* Filter Tabs */}
          <AnimatePresence>
            {hasAnalyzed && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
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
          </AnimatePresence>

          {/* Loading State */}
          <AnimatePresence>
            {isLoading && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="flex flex-col items-center justify-center py-32"
              >
                <div className="relative">
                  <LoadingSpinner
                    variant="ripple"
                    size="xl"
                    text={loadingPhase}
                    showProgress={true}
                    progress={progress}
                  />
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Empty State */}
          <AnimatePresence>
            {!isLoading && hasAnalyzed && displayedImages.length === 0 && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
              >
                <EmptyState
                  title="No images found"
                  description="Try adjusting your filters or analyzing a different directory."
                />
              </motion.div>
            )}
          </AnimatePresence>

          {/* Images Grid */}
          <AnimatePresence>
            {!isLoading && displayedImages.length > 0 && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.6 }}
              >
                <MasonryGallery
                  images={displayedImages
                    .filter(
                      img => img && img.path && typeof img.path === "string"
                    )
                    .map(
                      img =>
                        `${BACKEND_URL}/api/image?path=${encodeURIComponent(
                          img.path
                        )}`
                    )}
                  aspectRatio="auto"
                  showThumbnails={true}
                  enableZoom={true}
                  enableFullscreen={true}
                  loadingStrategy="lazy"
                  placeholderType="skeleton"
                />

                {/* Load More Trigger */}
                {hasMore && (
                  <div
                    ref={intersectionRef}
                    className="flex justify-center py-16"
                  >
                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="flex items-center gap-3 px-6 py-3 glass rounded-full border border-border/50"
                    >
                      <LoadingSpinner variant="dots" size="small" />
                      <span className="text-sm text-muted-foreground">
                        {isLoadingMore
                          ? "Loading more images..."
                          : "Scroll for more"}
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
                      <span className="text-sm text-muted-foreground">
                        You&apos;ve reached the end of the collection
                      </span>
                    </div>
                  </motion.div>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </main>

        {/* Settings Panel */}
        <SettingsPanel
          settings={settings}
          onSettingsChange={setSettings}
          open={isSettingsOpen}
          onOpenChange={setIsSettingsOpen}
        />

        {/* Toast Notifications */}
        <Toaster />
      </div>
    </ThemeProvider>
  );
}
