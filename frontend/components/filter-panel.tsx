"use client";

import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import { X } from "lucide-react";

interface ClusterData {
  id: number;
  size: number;
}

interface FilterPanelProps {
  isOpen: boolean;
  clusters: ClusterData[];
  selectedCluster: string;
  onClusterChange: (cluster: string) => void;
  hasDuplicates?: boolean;
  showDuplicates?: boolean;
  onShowDuplicates?: (show: boolean) => void;
}

export function FilterPanel({
  isOpen,
  clusters,
  selectedCluster,
  onClusterChange,
  hasDuplicates = false,
  showDuplicates = false,
  onShowDuplicates,
}: FilterPanelProps) {
  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: "auto" }}
          exit={{ opacity: 0, height: 0 }}
          transition={{ duration: 0.3 }}
          className="bg-card/50 backdrop-blur-sm border rounded-xl p-6 space-y-4"
        >
          <div className="flex items-center justify-between">
            <h3 className="font-semibold">Filter by Cluster</h3>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => onClusterChange("all")}
            >
              <X className="w-4 h-4 mr-2" />
              Clear
            </Button>
          </div>

          <div className="flex flex-wrap gap-2">
            <Button
              variant={
                selectedCluster === "all" && !showDuplicates
                  ? "default"
                  : "outline"
              }
              size="sm"
              className={`hover:bg-primary/20 transition-colors ${
                showDuplicates ? "opacity-50 pointer-events-none" : ""
              }`}
              onClick={() => {
                if (!showDuplicates) onClusterChange("all");
              }}
            >
              All Images (
              {clusters.reduce((sum, cluster) => sum + cluster.size, 0)})
            </Button>

            {clusters.map((cluster) => (
              <Button
                key={cluster.id}
                variant={
                  selectedCluster === cluster.id.toString() && !showDuplicates
                    ? "default"
                    : "outline"
                }
                size="sm"
                className={`hover:bg-primary/20 transition-colors ${
                  showDuplicates ? "opacity-50 pointer-events-none" : ""
                }`}
                onClick={() => {
                  if (!showDuplicates) onClusterChange(cluster.id.toString());
                }}
              >
                Cluster {cluster.id} ({cluster.size})
              </Button>
            ))}

            {hasDuplicates && onShowDuplicates && (
              <Button
                variant={showDuplicates ? "destructive" : "outline"}
                size="sm"
                className="hover:bg-primary/20 transition-colors"
                onClick={() => onShowDuplicates(!showDuplicates)}
              >
                Duplicates
              </Button>
            )}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
