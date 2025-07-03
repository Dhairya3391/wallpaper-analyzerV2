"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import { X, Filter, Sun, Star, Copy, Tag } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";

interface ClusterData {
  id: number;
  size: number;
  label?: string;
}

interface FilterPanelProps {
  isOpen: boolean;
  clusters: ClusterData[];
  selectedCluster: string;
  onClusterChange: (cluster: string) => void;
  hasDuplicates?: boolean;
  showDuplicates?: boolean;
  onShowDuplicates?: (show: boolean) => void;
  onLabelFilterChange?: (labels: string[]) => void;
  onScoreRangeChange?: (min: number, max: number) => void;
  onBrightnessRangeChange?: (min: number, max: number) => void;
}

export function FilterPanel({
  isOpen,
  clusters,
  selectedCluster,
  onClusterChange,
  hasDuplicates = false,
  showDuplicates = false,
  onShowDuplicates = () => {},
  onLabelFilterChange = () => {},
  onScoreRangeChange = () => {},
  onBrightnessRangeChange = () => {},
}: FilterPanelProps) {
  const [labelFilter, setLabelFilter] = useState<string[]>([]);
  const [scoreRange, setScoreRange] = useState<[number, number]>([0, 1]);
  const [brightnessRange, setBrightnessRange] = useState<[number, number]>([
    0, 1,
  ]);

  // Unique labels for chips
  const uniqueLabels = Array.from(
    new Set(
      (clusters || [])
        .map((c) => c.label)
        .filter((l) => l && l !== "unknown" && l !== "Unlabeled"),
    ),
  ) as string[];

  const handleLabelChip = (label: string) => {
    let newLabels: string[];
    if (labelFilter.includes(label)) {
      newLabels = labelFilter.filter((l) => l !== label);
    } else {
      newLabels = [...labelFilter, label];
    }
    setLabelFilter(newLabels);
    onLabelFilterChange(newLabels);
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          key="filter-panel"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 20 }}
          transition={{ duration: 0.3 }}
          className="bg-card/80 backdrop-blur border rounded-2xl p-6 space-y-6 shadow-xl sticky top-6 z-30 min-w-[320px] max-w-xs"
        >
          <div className="flex items-center justify-between mb-2 sticky top-0 z-10 bg-card/80 backdrop-blur rounded-t-2xl pb-2">
            <div className="flex items-center gap-2">
              <Filter className="w-5 h-5 text-primary" />
              <h3 className="font-semibold text-lg">Filters</h3>
            </div>
            <Button
              variant="ghost"
              size="icon"
              className="rounded-full hover:bg-destructive/10"
              onClick={() => {
                onClusterChange("all");
                setLabelFilter([]);
                setScoreRange([0, 1]);
                setBrightnessRange([0, 1]);
                onShowDuplicates(false);
                onLabelFilterChange([]);
                onScoreRangeChange(0, 1);
                onBrightnessRangeChange(0, 1);
              }}
              aria-label="Clear filters"
            >
              <X className="w-4 h-4" />
            </Button>
          </div>

          {/* Cluster Dropdown */}
          <div className="flex flex-col gap-2 p-3 rounded-xl bg-muted/30 border">
            <div className="flex items-center gap-2 mb-1">
              <Copy className="w-4 h-4 text-muted-foreground" />
              <span className="text-sm font-medium">Cluster</span>
            </div>
            <Select value={selectedCluster} onValueChange={onClusterChange}>
              <SelectTrigger>
                <SelectValue placeholder="All Clusters" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Clusters</SelectItem>
                {(clusters || []).map((cluster) => (
                  <SelectItem
                    key={cluster.id ?? "unknown"}
                    value={String(cluster.id ?? "unknown")}
                  >
                    {cluster.label &&
                    cluster.label !== "unknown" &&
                    cluster.label !== "Unlabeled"
                      ? `${cluster.label} (Cluster ${cluster.id ?? "?"})`
                      : `Cluster ${cluster.id ?? "?"}`}{" "}
                    {`(${cluster.size ?? 0})`}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Label Multi-Select Chips */}
          {uniqueLabels.length > 0 && (
            <div className="flex flex-col gap-2 p-3 rounded-xl bg-muted/30 border">
              <div className="flex items-center gap-2 mb-1">
                <Tag className="w-4 h-4 text-muted-foreground" />
                <span className="text-sm font-medium">Labels</span>
              </div>
              <div className="flex flex-wrap gap-2">
                {uniqueLabels.map((label) => (
                  <Badge
                    key={label}
                    variant={
                      labelFilter.includes(label) ? "default" : "outline"
                    }
                    className={`cursor-pointer transition-colors ${
                      labelFilter.includes(label)
                        ? "bg-primary text-primary-foreground"
                        : "hover:bg-muted/60"
                    }`}
                    onClick={() => handleLabelChip(label)}
                  >
                    {label}
                  </Badge>
                ))}
              </div>
            </div>
          )}

          {/* Score Range Slider */}
          <div className="flex flex-col gap-2 p-3 rounded-xl bg-muted/30 border">
            <div className="flex items-center gap-2 mb-1">
              <Star className="w-4 h-4 text-muted-foreground" />
              <span className="text-sm font-medium">Aesthetic Score</span>
              <span className="ml-auto text-xs text-muted-foreground">
                {scoreRange[0].toFixed(2)} - {scoreRange[1].toFixed(2)}
              </span>
            </div>
            <Slider
              min={0}
              max={1}
              step={0.01}
              value={scoreRange}
              onValueChange={(val) => {
                setScoreRange(val as [number, number]);
                onScoreRangeChange(val[0], val[1]);
              }}
              className="w-full"
            />
          </div>

          {/* Brightness Range Slider */}
          <div className="flex flex-col gap-2 p-3 rounded-xl bg-muted/30 border">
            <div className="flex items-center gap-2 mb-1">
              <Sun className="w-4 h-4 text-muted-foreground" />
              <span className="text-sm font-medium">Brightness</span>
              <span className="ml-auto text-xs text-muted-foreground">
                {brightnessRange[0].toFixed(2)} -{" "}
                {brightnessRange[1].toFixed(2)}
              </span>
            </div>
            <Slider
              min={0}
              max={1}
              step={0.01}
              value={brightnessRange}
              onValueChange={(val) => {
                setBrightnessRange(val as [number, number]);
                onBrightnessRangeChange(val[0], val[1]);
              }}
              className="w-full"
            />
          </div>

          {/* Duplicates Filter */}
          {hasDuplicates && (
            <div className="flex flex-col gap-2 p-3 rounded-xl bg-muted/30 border">
              <div className="flex items-center gap-2 mb-1">
                <Copy className="w-4 h-4 text-muted-foreground" />
                <span className="text-sm font-medium">Duplicates</span>
              </div>
              <Select
                value={showDuplicates ? "duplicates" : "all"}
                onValueChange={(val) => onShowDuplicates(val === "duplicates")}
              >
                <SelectTrigger>
                  <SelectValue placeholder="All Images" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Images</SelectItem>
                  <SelectItem value="duplicates">
                    Show Duplicates Only
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>
          )}
        </motion.div>
      )}
    </AnimatePresence>
  );
}
