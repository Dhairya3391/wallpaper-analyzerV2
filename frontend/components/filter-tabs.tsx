"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Search, X } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";

interface ClusterData {
  id: number;
  size: number;
}

interface FilterTabsProps {
  clusters: ClusterData[];
  selectedCluster: string;
  onClusterChange: (cluster: string) => void;
  hasDuplicates: boolean;
  showDuplicates: boolean;
  onShowDuplicates: (show: boolean) => void;
  searchTerm: string;
  onSearchChange: (term: string) => void;
  totalImages: number;
  filteredCount: number;
}

export function FilterTabs({
  clusters,
  selectedCluster,
  onClusterChange,
  hasDuplicates,
  showDuplicates,
  onShowDuplicates,
  searchTerm,
  onSearchChange,
  totalImages,
  filteredCount,
}: FilterTabsProps) {
  const [isSearchFocused, setIsSearchFocused] = useState(false);

  const handleTabClick = (value: string) => {
    if (value === "duplicates") {
      onShowDuplicates(!showDuplicates);
      if (!showDuplicates) {
        onClusterChange("all");
      }
    } else {
      onClusterChange(value);
      if (showDuplicates) {
        onShowDuplicates(false);
      }
    }
  };

  return (
    <div className="space-y-6">
      {/* Search Bar */}
      <div className="flex justify-center">
        <div
          className={`relative w-full max-w-md transition-all duration-300 ${
            isSearchFocused ? "max-w-lg" : ""
          }`}
        >
          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
            <Search className="h-5 w-5 text-gray-400" />
          </div>
          <Input
            type="text"
            placeholder="Search images..."
            value={searchTerm}
            onChange={(e) => onSearchChange(e.target.value)}
            onFocus={() => setIsSearchFocused(true)}
            onBlur={() => setIsSearchFocused(false)}
            className="pl-10 pr-10 py-3 rounded-full border-gray-200 focus:border-blue-500 focus:ring-blue-500"
          />
          {searchTerm && (
            <button
              onClick={() => onSearchChange("")}
              className="absolute inset-y-0 right-0 pr-3 flex items-center"
            >
              <X className="h-5 w-5 text-gray-400 hover:text-gray-600" />
            </button>
          )}
        </div>
      </div>

      {/* Filter Tabs */}
      <div className="flex flex-wrap justify-center gap-3">
        <motion.div
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <Badge
            variant={selectedCluster === "all" && !showDuplicates ? "default" : "outline"}
            className={`cursor-pointer px-4 py-2 text-sm font-medium transition-all duration-200 ${
              selectedCluster === "all" && !showDuplicates
                ? "bg-blue-600 text-white hover:bg-blue-700"
                : "hover:bg-gray-100"
            }`}
            onClick={() => handleTabClick("all")}
          >
            All Images ({totalImages})
          </Badge>
        </motion.div>

        {clusters.map((cluster) => (
          <motion.div
            key={cluster.id}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <Badge
              variant={
                selectedCluster === cluster.id.toString() && !showDuplicates
                  ? "default"
                  : "outline"
              }
              className={`cursor-pointer px-4 py-2 text-sm font-medium transition-all duration-200 ${
                selectedCluster === cluster.id.toString() && !showDuplicates
                  ? "bg-blue-600 text-white hover:bg-blue-700"
                  : "hover:bg-gray-100"
              }`}
              onClick={() => handleTabClick(cluster.id.toString())}
            >
              Cluster {cluster.id} ({cluster.size})
            </Badge>
          </motion.div>
        ))}

        {hasDuplicates && (
          <motion.div
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <Badge
              variant={showDuplicates ? "default" : "outline"}
              className={`cursor-pointer px-4 py-2 text-sm font-medium transition-all duration-200 ${
                showDuplicates
                  ? "bg-red-600 text-white hover:bg-red-700"
                  : "hover:bg-gray-100"
              }`}
              onClick={() => handleTabClick("duplicates")}
            >
              Duplicates
            </Badge>
          </motion.div>
        )}
      </div>

      {/* Results Count */}
      {filteredCount !== totalImages && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center text-gray-600"
        >
          Showing {filteredCount} of {totalImages} images
        </motion.div>
      )}
    </div>
  );
}