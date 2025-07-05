"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Search, Settings, Folder } from "lucide-react";
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
        className={`relative flex items-center bg-white border-2 rounded-full shadow-lg transition-all duration-300 ${
          isFocused
            ? "border-blue-500 shadow-xl"
            : "border-gray-200 hover:border-gray-300"
        }`}
      >
        <div className="flex items-center pl-6">
          <Folder className="w-5 h-5 text-gray-400" />
        </div>
        
        <Input
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onFocus={() => setIsFocused(true)}
          onBlur={() => setIsFocused(false)}
          placeholder={placeholder}
          className="flex-1 border-0 bg-transparent px-4 py-4 text-lg focus:ring-0 focus:outline-none"
        />

        <Button
          onClick={onAnalyze}
          disabled={isLoading || !value.startsWith("/")}
          className="mr-2 rounded-full bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 transition-all duration-200"
        >
          {isLoading ? (
            <div className="flex items-center">
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
              Analyzing...
            </div>
          ) : (
            <div className="flex items-center">
              <Search className="w-4 h-4 mr-2" />
              Analyze
            </div>
          )}
        </Button>
      </div>

      {value && !value.startsWith("/") && (
        <motion.p
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-sm text-amber-600 mt-2 text-center"
        >
          Please enter an absolute path (starting with /)
        </motion.p>
      )}
    </motion.div>
  );
}