"use client";

import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Folder, Sparkles, X } from "lucide-react";
import { Input } from "@/components/ui/input";
import { LoadingSpinner } from "@/components/loading-spinner";
import { Button } from "@/components/ui/button";
import { useIsMobile } from "@/hooks/use-mobile";

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
  const isMobile = useIsMobile();
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

    setIsTyping(true);
    if (typingTimeoutRef.current) clearTimeout(typingTimeoutRef.current);
    typingTimeoutRef.current = setTimeout(() => setIsTyping(false), 500);
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
      className={`${isMobile ? "max-w-full" : "max-w-2xl"} mx-auto`}
    >
      <div
        className={`relative flex ${isMobile ? "flex-col gap-3" : "items-center gap-2"} px-4 py-3 rounded-2xl border-2 glass shadow-medium transition-all duration-300 ${
          isFocused
            ? "border-primary/50 shadow-strong"
            : "border-border/50 hover:border-border"
        } ${showError ? "border-destructive/50" : ""}`}
      >
        {/* Icon */}
        <Folder
          className={`w-5 h-5 transition-colors duration-200 ${
            isFocused ? "text-primary" : "text-muted-foreground"
          }`}
        />

        {/* Input */}
        <div className="relative flex-1">
          <Input
            ref={inputRef}
            value={value}
            onChange={handleInputChange}
            onFocus={() => setIsFocused(true)}
            onBlur={() => setIsFocused(false)}
            onKeyPress={handleKeyPress}
            placeholder={placeholder}
            className={`border-none bg-transparent focus:ring-0 focus:outline-none placeholder:text-muted-foreground/60 px-2 py-1 ${isMobile ? "text-base" : ""}`}
          />
          {/* Typing Indicator */}
          <AnimatePresence>
            {isTyping && (
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                className="absolute right-2 top-1/2 transform -translate-y-1/2"
              >
                <LoadingSpinner variant="dots" size="small" />
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Clear Button */}
        <AnimatePresence>
          {showClearButton && value && !isLoading && (
            <motion.button
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.8 }}
              onClick={handleClear}
              className="p-2 rounded-lg hover:bg-muted/50 transition-colors"
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.95 }}
            >
              <X className="w-4 h-4 text-muted-foreground" />
            </motion.button>
          )}
        </AnimatePresence>

        {/* Analyze Button */}
        <Button
          onClick={onAnalyze}
          disabled={isLoading || !isValidPath}
          loading={isLoading}
          loadingText="Analyzing..."
          className={`rounded-xl btn-primary ${isMobile ? "w-full px-4 py-3" : "px-6 py-2"} disabled:opacity-50 disabled:cursor-not-allowed`}
          ripple
        >
          {!isLoading && (
            <div className="flex items-center justify-center">
              <Sparkles className="w-4 h-4 mr-2" />
              Analyze
            </div>
          )}
        </Button>
      </div>

      {/* Error Message */}
      <AnimatePresence>
        {showError && (
          <motion.div
            initial={{ opacity: 0, y: -10, height: 0 }}
            animate={{ opacity: 1, y: 0, height: "auto" }}
            exit={{ opacity: 0, y: -10, height: 0 }}
            transition={{ duration: 0.3 }}
            className="mt-4 p-4 glass border border-destructive/20 rounded-xl"
          >
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-destructive animate-pulse" />
              <p className="text-sm text-destructive">
                💡 Please enter an absolute path (starting with /)
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Success Message */}
      <AnimatePresence>
        {isValidPath && value && !isLoading && (
          <motion.div
            initial={{ opacity: 0, y: -10, height: 0 }}
            animate={{ opacity: 1, y: 0, height: "auto" }}
            exit={{ opacity: 0, y: -10, height: 0 }}
            transition={{ duration: 0.3 }}
            className="mt-4 p-4 glass border border-green-500/20 rounded-xl"
          >
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-green-500" />
              <p className="text-sm text-green-600">✅ Valid path detected</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
