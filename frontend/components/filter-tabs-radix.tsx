"use client";

import { useState } from "react";
import { Search, X, Copy, TrendingUp, Filter } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";

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

  const handleValueChange = (value: string) => {
    if (value === "duplicates") {
      onShowDuplicates(!showDuplicates);
      if (!showDuplicates) onClusterChange("all");
    } else {
      onClusterChange(value);
      if (showDuplicates) onShowDuplicates(false);
    }
  };

  const currentValue = showDuplicates ? "duplicates" : selectedCluster;

  return (
    <div className="space-y-4">
      {/* Search Bar */}
      <div className="flex justify-center">
        <div
          className={`relative w-full max-w-md transition-all duration-300 ${
            isSearchFocused ? "max-w-lg" : ""
          }`}
        >
          <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
            <Search
              className={`h-5 w-5 transition-colors duration-200 ${
                isSearchFocused ? "text-primary" : "text-muted-foreground"
              }`}
            />
          </div>
          <Input
            type="text"
            placeholder="Search images..."
            value={searchTerm}
            onChange={e => onSearchChange(e.target.value)}
            onFocus={() => setIsSearchFocused(true)}
            onBlur={() => setIsSearchFocused(false)}
            className="pl-12 pr-12 py-3 rounded-2xl glass border-border/50 focus:border-primary/50 focus:ring-primary/20 transition-all duration-200 placeholder:text-muted-foreground/60"
          />
          {searchTerm && (
            <button
              onClick={() => onSearchChange("")}
              className="absolute inset-y-0 right-0 pr-4 flex items-center hover:bg-muted/50 rounded-r-2xl transition-colors duration-200"
            >
              <X className="h-5 w-5 text-muted-foreground hover:text-foreground" />
            </button>
          )}
        </div>
      </div>

      {/* Filter Header */}
      <div className="flex items-center justify-center gap-2 text-professional-muted">
        <Filter className="w-4 h-4" />
        <span className="text-sm font-medium">Filter by category</span>
      </div>

      {/* Tabs */}
      <Tabs
        value={currentValue}
        onValueChange={handleValueChange}
        className="w-full"
      >
        <TabsList className="flex flex-wrap justify-center gap-3 bg-transparent p-0">
          <TabsTrigger
            value="all"
            className={`px-6 py-3 text-sm font-medium rounded-xl transition-all duration-300 ${
              currentValue === "all" && !showDuplicates
                ? "btn-primary shadow-medium"
                : "glass hover:bg-muted/50 border border-border/50"
            }`}
          >
            <TrendingUp className="w-4 h-4 mr-2" />
            All Images ({totalImages})
          </TabsTrigger>

          {clusters.map(cluster => (
            <TabsTrigger
              key={cluster.id}
              value={cluster.id.toString()}
              className={`px-6 py-3 text-sm font-medium rounded-xl transition-all duration-300 ${
                currentValue === cluster.id.toString() && !showDuplicates
                  ? "btn-primary shadow-medium"
                  : "glass hover:bg-muted/50 border border-border/50"
              }`}
            >
              <div className="w-2 h-2 rounded-full bg-current mr-2" />
              Cluster {cluster.id} ({cluster.size})
            </TabsTrigger>
          ))}

          {hasDuplicates && (
            <TabsTrigger
              value="duplicates"
              className={`px-6 py-3 text-sm font-medium rounded-xl transition-all duration-300 ${
                showDuplicates
                  ? "bg-destructive text-destructive-foreground shadow-medium"
                  : "glass hover:bg-muted/50 border border-border/50"
              }`}
            >
              <Copy className="w-4 h-4 mr-2" />
              Duplicates
            </TabsTrigger>
          )}
        </TabsList>
      </Tabs>

      {/* Results Count */}
      {filteredCount !== totalImages && (
        <div className="text-center">
          <div className="inline-flex items-center gap-2 px-4 py-2 glass rounded-full border border-border/50">
            <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />
            <span className="text-sm text-professional-muted">
              Showing{" "}
              <span className="font-semibold text-professional">
                {filteredCount}
              </span>{" "}
              of{" "}
              <span className="font-semibold text-professional">
                {totalImages}
              </span>{" "}
              images
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
