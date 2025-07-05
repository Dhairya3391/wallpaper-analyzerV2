"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Search, Folder, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

interface SearchBarProps {
  value: string;
  onChange: (value: string) => void;
  onAnalyze: () => void;
  isLoading: boolean;
  placeholder?: string;
}

export function SearchBar({
  value,
  onChange,
  onAnalyze,
  isLoading,
  placeholder = "Search...",
}: SearchBarProps) {
  const [isFocused, setIsFocused] = useState(false);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.2 }}
      className="max-w-2xl mx-auto"
    >
      <div
        className={`relative flex items-center glass border-2 rounded-2xl shadow-luxury transition-all duration-300 ${
          isFocused
            ? "border-primary/50 shadow-glow"
            : "border-border/50 hover:border-border"
        }`}
      >
        <div className="flex items-center pl-6">
          <Folder className={`w-5 h-5 transition-colors duration-200 ${
            isFocused ? "text-primary" : "text-muted-foreground"
          }`} />
        </div>
        
        <Input
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onFocus={() => setIsFocused(true)}
          onBlur={() => setIsFocused(false)}
          placeholder={placeholder}
          className="flex-1 border-0 bg-transparent px-4 py-4 text-lg focus:ring-0 focus:outline-none placeholder:text-muted-foreground/60"
        />

        <Button
          onClick={onAnalyze}
          disabled={isLoading || !value.startsWith("/")}
          className="mr-2 rounded-xl gradient-primary hover:shadow-glow text-white px-6 py-2 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed btn-gradient"
        >
          {isLoading ? (
            <div className="flex items-center">
              <motion.div 
                className="w-4 h-4 border-2 border-white border-t-transparent rounded-full mr-2"
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
              />
              Analyzing...
            </div>
          ) : (
            <div className="flex items-center">
              <Sparkles className="w-4 h-4 mr-2" />
              Analyze
            </div>
          )}
        </Button>
      </div>

      {value && !value.startsWith("/") && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-3 p-3 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg"
        >
          <p className="text-sm text-amber-700 dark:text-amber-300 text-center">
            💡 Please enter an absolute path (starting with /)
          </p>
        </motion.div>
      )}
    </motion.div>
  );
}