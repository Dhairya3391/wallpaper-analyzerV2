"use client";

import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Folder, Sparkles, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { LoadingSpinner } from "@/components/loading-spinner";

interface SearchBarProps {
  value: string;
  onChange: (value: string) => void;
  onAnalyze: () => void;
  isLoading: boolean;
  placeholder?: string;
  showClearButton?: boolean;
  autoFocus?: boolean;
}

export function SearchBar({
  value,
  onChange,
  onAnalyze,
  isLoading,
  placeholder = "Search...",
  showClearButton = true,
  autoFocus = false,
}: SearchBarProps) {
  const [isFocused, setIsFocused] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const typingTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    if (autoFocus && inputRef.current) {
      inputRef.current.focus();
    }
  }, [autoFocus]);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value;
    onChange(newValue);

    // Show typing indicator
    setIsTyping(true);

    // Clear previous timeout
    if (typingTimeoutRef.current) {
      clearTimeout(typingTimeoutRef.current);
    }

    // Hide typing indicator after delay
    typingTimeoutRef.current = setTimeout(() => {
      setIsTyping(false);
    }, 500);
  };

  const handleClear = () => {
    onChange("");
    inputRef.current?.focus();
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && value.startsWith("/") && !isLoading) {
      onAnalyze();
    }
  };

  const isValidPath = value.startsWith("/");
  const showError = value && !isValidPath;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.2, duration: 0.6 }}
      className="max-w-2xl mx-auto"
    >
      <div
        className={`relative flex items-center glass border-2 rounded-2xl shadow-medium transition-all duration-300 ${
          isFocused
            ? "border-primary/50 shadow-strong"
            : "border-border/50 hover:border-border"
        } ${showError ? "border-destructive/50" : ""}`}
      >
        <motion.div
          className="flex items-center pl-6"
          animate={{ scale: isFocused ? 1.1 : 1 }}
          transition={{ duration: 0.2 }}
        >
          <Folder
            className={`w-5 h-5 transition-colors duration-200 ${
              isFocused ? "text-primary" : "text-muted-foreground"
            }`}
          />
        </motion.div>

        <div className="flex-1 relative">
          <Input
            ref={inputRef}
            type="text"
            value={value}
            onChange={handleInputChange}
            onFocus={() => setIsFocused(true)}
            onBlur={() => setIsFocused(false)}
            onKeyPress={handleKeyPress}
            placeholder={placeholder}
            className="border-0 bg-transparent px-4 py-4 text-lg focus:ring-0 focus:outline-none placeholder:text-muted-foreground/60"
          />

          {/* Typing indicator */}
          <AnimatePresence>
            {isTyping && (
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                className="absolute right-4 top-1/2 transform -translate-y-1/2"
              >
                <LoadingSpinner variant="dots" size="small" />
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Clear button */}
        <AnimatePresence>
          {showClearButton && value && !isLoading && (
            <motion.button
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.8 }}
              onClick={handleClear}
              className="mr-2 p-2 rounded-lg hover:bg-muted/50 transition-colors duration-200"
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.95 }}
            >
              <X className="w-4 h-4 text-muted-foreground" />
            </motion.button>
          )}
        </AnimatePresence>

        <Button
          onClick={onAnalyze}
          disabled={isLoading || !isValidPath}
          loading={isLoading}
          loadingText="Analyzing..."
          className="mr-2 rounded-xl btn-primary px-6 py-2 disabled:opacity-50 disabled:cursor-not-allowed"
          ripple={true}
        >
          {!isLoading && (
            <div className="flex items-center">
              <Sparkles className="w-4 h-4 mr-2" />
              Analyze
            </div>
          )}
        </Button>
      </div>

      {/* Error message */}
      <AnimatePresence>
        {showError && (
          <motion.div
            initial={{ opacity: 0, y: -10, height: 0 }}
            animate={{ opacity: 1, y: 0, height: "auto" }}
            exit={{ opacity: 0, y: -10, height: 0 }}
            transition={{ duration: 0.3 }}
            className="mt-4 p-4 glass border border-destructive/20 rounded-xl overflow-hidden"
          >
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-destructive animate-pulse" />
              <p className="text-sm text-destructive text-center">
                💡 Please enter an absolute path (starting with /)
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Success indicator */}
      <AnimatePresence>
        {isValidPath && value && !isLoading && (
          <motion.div
            initial={{ opacity: 0, y: -10, height: 0 }}
            animate={{ opacity: 1, y: 0, height: "auto" }}
            exit={{ opacity: 0, y: -10, height: 0 }}
            transition={{ duration: 0.3 }}
            className="mt-4 p-4 glass border border-green-500/20 rounded-xl overflow-hidden"
          >
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-green-500" />
              <p className="text-sm text-green-600 text-center">
                ✅ Valid path detected
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
