"use client";

import { motion } from "framer-motion";
import { Images, Layers, Sparkles, TrendingUp } from "lucide-react";

interface StatsBarProps {
  totalImages: number;
  filteredImages: number;
  clusters: number;
}

export function StatsBar({
  totalImages,
  filteredImages,
  clusters,
}: StatsBarProps) {
  const stats = [
    {
      icon: Images,
      label: "Total Images",
      value: totalImages.toLocaleString(),
      color: "text-blue-500",
      bgColor: "bg-blue-500/10",
    },
    {
      icon: TrendingUp,
      label: "Showing",
      value: filteredImages.toLocaleString(),
      color: "text-green-500",
      bgColor: "bg-green-500/10",
    },
    {
      icon: Layers,
      label: "Clusters",
      value: clusters.toString(),
      color: "text-purple-500",
      bgColor: "bg-purple-500/10",
    },
    {
      icon: Sparkles,
      label: "Quality Score",
      value: "85%",
      color: "text-yellow-500",
      bgColor: "bg-yellow-500/10",
    },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="grid grid-cols-2 lg:grid-cols-4 gap-4"
    >
      {stats.map((stat, index) => (
        <motion.div
          key={stat.label}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: index * 0.1 }}
          className="bg-card/50 backdrop-blur-sm border rounded-xl p-4 hover:bg-card/70 transition-colors duration-300"
        >
          <div className="flex items-center gap-3">
            <div className={`p-2 rounded-lg ${stat.bgColor}`}>
              <stat.icon className={`w-5 h-5 ${stat.color}`} />
            </div>
            <div>
              <p className="text-2xl font-bold">{stat.value}</p>
              <p className="text-sm text-muted-foreground">{stat.label}</p>
            </div>
          </div>
        </motion.div>
      ))}
    </motion.div>
  );
}
