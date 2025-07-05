"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Search, X, Filter, TrendingUp } from "lucide-react";
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
    <div className="space-y-8">
      {/* Search Bar */}
      <div className="flex justify-center">
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.3 }}
          className={`relative w-full max-w-md transition-all duration-300 ${
            isSearchFocused ? "max-w-lg" : ""
          }`}
        >
          <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
            <Search className={`h-5 w-5 transition-colors duration-200 ${
              isSearchFocused ? "text-primary" : "text-muted-foreground"
            }`} />
          </div>
          <Input
            type="text"
            placeholder="Search images..."
            value={searchTerm}
            onChange={(e) => onSearchChange(e.target.value)}
            onFocus={() => setIsSearchFocused(true)}
            onBlur={() => setIsSearchFocused(false)}
            className="pl-12 pr-12 py-3 rounded-2xl glass border-border/50 focus:border-primary/50 focus:ring-primary/20 transition-all duration-200 placeholder:text-muted-foreground/60"
          />
          {searchTerm && (
            <motion.button
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              onClick={() => onSearchChange("")}
              className="absolute inset-y-0 right-0 pr-4 flex items-center hover:bg-muted/50 rounded-r-2xl transition-colors duration-200"
            >
              <X className="h-5 w-5 text-muted-foreground hover:text-foreground" />
            </motion.button>
          )}
        </motion.div>
      </div>

      {/* Filter Header */}
      <div className="flex items-center justify-center gap-2 text-muted-foreground">
        <Filter className="w-4 h-4" />
        <span className="text-sm font-medium">Filter by category</span>
      </div>

      {/* Filter Tabs */}
      <div className="flex flex-wrap justify-center gap-3">
        <motion.div
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          transition={{ type: "spring", stiffness: 400, damping: 17 }}
        >
          <Badge
            variant={selectedCluster === "all" && !showDuplicates ? "default" : "outline"}
            className={`cursor-pointer px-6 py-3 text-sm font-medium transition-all duration-300 rounded-xl ${
              selectedCluster === "all" && !showDuplicates
                ? "gradient-primary text-white shadow-glow hover:shadow-luxury"
                : "glass hover:bg-muted/50 border-border/50"
            }`}
            onClick={() => handleTabClick("all")}
          >
            <TrendingUp className="w-4 h-4 mr-2" />
            All Images ({totalImages})
          </Badge>
        </motion.div>

        {clusters.map((cluster, index) => (
          <motion.div
            key={cluster.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: index * 0.05 }}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <Badge
              variant={
                selectedCluster === cluster.id.toString() && !showDuplicates
                  ? "default"
                  : "outline"
              }
              className={`cursor-pointer px-6 py-3 text-sm font-medium transition-all duration-300 rounded-xl ${
                selectedCluster === cluster.id.toString() && !showDuplicates
                  ? "gradient-primary text-white shadow-glow hover:shadow-luxury"
                  : "glass hover:bg-muted/50 border-border/50"
              }`}
              onClick={() => handleTabClick(cluster.id.toString())}
            >
              <div className="w-2 h-2 rounded-full bg-current mr-2" />
              Cluster {cluster.id} ({cluster.size})
            </Badge>
          </motion.div>
        ))}

        {hasDuplicates && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: clusters.length * 0.05 }}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <Badge
              variant={showDuplicates ? "default" : "outline"}
              className={`cursor-pointer px-6 py-3 text-sm font-medium transition-all duration-300 rounded-xl ${
                showDuplicates
                  ? "bg-gradient-to-r from-red-500 to-pink-500 text-white shadow-glow hover:shadow-luxury"
                  : "glass hover:bg-muted/50 border-border/50"
              }`}
              onClick={() => handleTabClick("duplicates")}
            >
              <div className="w-2 h-2 rounded-full bg-current mr-2" />
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
          className="text-center"
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 glass rounded-full border border-border/50">
            <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />
            <span className="text-sm text-muted-foreground">
              Showing <span className="font-semibold text-foreground">{filteredCount}</span> of{" "}
              <span className="font-semibold text-foreground">{totalImages}</span> images
            </span>
          </div>
        </motion.div>
      )}
    </div>
  );
}