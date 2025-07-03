"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Header } from "@/components/header";
import { ImageMasonry } from "@/components/image-masonry";
import { StatsBar } from "@/components/stats-bar";
import { FilterPanel } from "@/components/filter-panel";
import { SettingsPanel } from "@/components/settings-panel";
import { ThemeProvider } from "@/components/theme-provider";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

// Define ImageData type
interface ImageData {
  path: string;
  cluster?: number;
  cluster_size?: number;
  is_duplicate?: boolean;
  is_low_aesthetic?: boolean;
  aesthetic_score?: number;
}

export default function Home() {
  const [images, setImages] = useState<ImageData[]>([]);
  const [filteredImages, setFilteredImages] = useState<ImageData[]>([]);
  const [clusters, setClusters] = useState<{ id: number; size: number }[]>([]);
  const [selectedCluster, setSelectedCluster] = useState("all");
  const [showDuplicates, setShowDuplicates] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showTip, setShowTip] = useState(false);

  const [settings, setSettings] = useState({
    directory: "/Users/dhairya/Downloads/walls",
    similarity_threshold: 0.85,
    aesthetic_threshold: 0.85,
    recursive: true,
    skip_duplicates: false,
    skip_aesthetics: false,
    limit: 1000,
  });

  const analyzeDirectory = async () => {
    setIsLoading(true);
    setError(null);
    setShowTip(false);
    const start = Date.now();
    try {
      const response = await fetch(`${BACKEND_URL}/api/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(settings),
      });
      const data = await response.json();
      if (!data.success) throw new Error(data.error || "Analysis failed");
      setImages(data.images);
      setFilteredImages(data.images);
      // Try to extract clusters from images (for cache or legacy data)
      let clustersArr: { id: number; size: number }[] = [];
      const clusterMap = new Map<number, number>();
      data.images.forEach((img: ImageData) => {
        const clusterId =
          typeof img.cluster === "number" ? img.cluster : undefined;
        if (clusterId !== undefined && clusterId !== -1) {
          clusterMap.set(clusterId, (clusterMap.get(clusterId) || 0) + 1);
        }
      });
      clustersArr = Array.from(clusterMap.entries()).map(([id, size]) => ({
        id,
        size,
      }));
      // If all cluster sizes are 0, try to use cluster_size from images (cache fallback)
      if (clustersArr.length === 0 && data.images.length > 0) {
        const fallbackMap = new Map<number, number>();
        data.images.forEach((img: ImageData) => {
          const clusterId =
            typeof img.cluster === "number" ? img.cluster : undefined;
          if (
            clusterId !== undefined &&
            clusterId !== -1 &&
            typeof img.cluster_size === "number"
          ) {
            fallbackMap.set(
              clusterId,
              Math.max(fallbackMap.get(clusterId) || 0, img.cluster_size),
            );
          }
        });
        clustersArr = Array.from(fallbackMap.entries()).map(([id, size]) => ({
          id,
          size,
        }));
      }
      setClusters(clustersArr);
      // Show tip if analysis took more than 10 seconds
      if (Date.now() - start > 10000) setShowTip(true);
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("Unknown error");
      }
      setImages([]);
      setFilteredImages([]);
      setClusters([]);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (showDuplicates) {
      setFilteredImages(images.filter((img) => img.is_duplicate));
    } else if (selectedCluster === "all") {
      setFilteredImages(images);
    } else {
      setFilteredImages(
        images.filter((img) => String(img.cluster) === selectedCluster),
      );
    }
  }, [selectedCluster, images, showDuplicates]);

  const totalImages = images.length;
  const totalClusters = clusters.length;
  const isAbsolutePath = settings.directory.startsWith("/");

  const hasDuplicates = images.some((img) => img.is_duplicate);

  return (
    <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
      <div className="flex flex-col min-h-screen bg-background text-foreground">
        <Header />

        <main className="flex-1 p-4 md:p-8 lg:p-12 transition-all duration-300 ease-in-out mx-auto w-full max-w-6xl">
          <div className="space-y-6 mb-8">
            <div className="flex flex-col md:flex-row md:items-end md:space-x-4 space-y-4 md:space-y-0">
              <div className="flex-1">
                <FilterPanel
                  isOpen={true}
                  clusters={clusters}
                  selectedCluster={selectedCluster}
                  onClusterChange={setSelectedCluster}
                  hasDuplicates={hasDuplicates}
                  showDuplicates={showDuplicates}
                  onShowDuplicates={setShowDuplicates}
                />
              </div>
              <div className="min-w-[260px] md:min-w-[320px]">
                <div className="bg-card/50 backdrop-blur-sm border rounded-xl p-6 flex flex-col gap-4">
                  <div>
                    <Input
                      value={settings.directory}
                      onChange={(e) =>
                        setSettings((s) => ({
                          ...s,
                          directory: e.target.value,
                        }))
                      }
                      placeholder="Directory path (e.g. /Users/yourname/Pictures)"
                      className="w-full"
                    />
                  </div>
                  <div className="flex gap-2">
                    <Button
                      onClick={analyzeDirectory}
                      variant="default"
                      className="flex-1"
                      disabled={isLoading || !isAbsolutePath}
                    >
                      {isLoading ? (
                        <span className="flex items-center justify-center">
                          <span className="animate-spin mr-2 w-4 h-4 border-2 border-current border-t-transparent rounded-full" />
                          Analyzing...
                        </span>
                      ) : (
                        "Analyze"
                      )}
                    </Button>
                    <Button
                      onClick={() => setIsSettingsOpen(true)}
                      variant="outline"
                      className="flex-shrink-0"
                      type="button"
                    >
                      Settings
                    </Button>
                  </div>
                </div>
              </div>
            </div>
            <div className="flex items-center gap-4 mt-2">
              {isLoading && (
                <span className="flex items-center text-xs text-muted-foreground">
                  <span className="animate-spin mr-2 w-4 h-4 border-2 border-current border-t-transparent rounded-full" />
                  Analyzing images...
                </span>
              )}
            </div>
            {!isAbsolutePath && (
              <div className="text-yellow-600 text-xs mt-2">
                Please enter an absolute directory path (e.g.
                /Users/yourname/Pictures)
              </div>
            )}
            {showTip && (
              <div className="text-blue-600 text-xs mt-2">
                Tip: For faster analysis, increase <b>MAX_WORKERS</b> and{" "}
                <b>BATCH_SIZE</b> in <code>wallpaper_analyzer.py</code>{" "}
                (currently set to {process.env.NEXT_PUBLIC_M1_WORKERS || 32}{" "}
                workers and batch size {process.env.NEXT_PUBLIC_M1_BATCH || 64}{" "}
                for M1 Macs).
              </div>
            )}
            {error && (
              <div className="text-red-500 text-sm mt-2" role="alert">
                {error}
                <Button
                  size="sm"
                  variant="outline"
                  className="ml-2"
                  onClick={analyzeDirectory}
                >
                  Retry
                </Button>
              </div>
            )}
          </div>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <StatsBar
              totalImages={totalImages}
              filteredImages={filteredImages.length}
              clusters={totalClusters}
            />
            <ImageMasonry images={filteredImages} isLoading={isLoading} />
          </motion.div>
        </main>

        <SettingsPanel
          settings={settings}
          onSettingsChange={setSettings}
          open={isSettingsOpen}
          onOpenChange={setIsSettingsOpen}
        />
      </div>
    </ThemeProvider>
  );
}
