"use client";

import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

interface LoadingSpinnerProps {
  size?: "small" | "medium" | "large" | "xl";
  variant?: "spinner" | "dots" | "pulse" | "bars" | "ripple";
  className?: string;
  text?: string;
  showProgress?: boolean;
  progress?: number;
}

export function LoadingSpinner({
  size = "medium",
  variant = "spinner",
  className = "",
  text,
  showProgress = false,
  progress = 0,
}: LoadingSpinnerProps) {
  const sizeClasses = {
    small: "w-4 h-4",
    medium: "w-8 h-8",
    large: "w-12 h-12",
    xl: "w-16 h-16",
  };

  const textSizes = {
    small: "text-xs",
    medium: "text-sm",
    large: "text-base",
    xl: "text-lg",
  };

  const renderSpinner = () => {
    switch (variant) {
      case "dots":
        return (
          <div className="flex space-x-1">
            {[0, 1, 2].map(i => (
              <motion.div
                key={i}
                className="w-2 h-2 bg-primary rounded-full"
                animate={{
                  scale: [1, 1.2, 1],
                  opacity: [0.5, 1, 0.5],
                }}
                transition={{
                  duration: 1.4,
                  repeat: Infinity,
                  delay: i * 0.2,
                  ease: "easeInOut",
                }}
              />
            ))}
          </div>
        );

      case "pulse":
        return (
          <motion.div
            className={cn("bg-primary rounded-full", sizeClasses[size])}
            animate={{
              scale: [1, 1.1, 1],
              opacity: [0.7, 1, 0.7],
            }}
            transition={{
              duration: 1.5,
              repeat: Infinity,
              ease: "easeInOut",
            }}
          />
        );

      case "bars":
        return (
          <div className="flex space-x-1">
            {[0, 1, 2, 3, 4].map(i => (
              <motion.div
                key={i}
                className="w-1 bg-primary rounded-full"
                style={{
                  height:
                    size === "small"
                      ? "12px"
                      : size === "medium"
                        ? "16px"
                        : "20px",
                }}
                animate={{
                  scaleY: [1, 2, 1],
                  opacity: [0.5, 1, 0.5],
                }}
                transition={{
                  duration: 1.2,
                  repeat: Infinity,
                  delay: i * 0.1,
                  ease: "easeInOut",
                }}
              />
            ))}
          </div>
        );

      case "ripple":
        return (
          <div className="relative">
            <motion.div
              className={cn(
                "border-2 border-primary rounded-full",
                sizeClasses[size]
              )}
              animate={{
                scale: [1, 1.2, 1],
                opacity: [1, 0.5, 1],
              }}
              transition={{
                duration: 2,
                repeat: Infinity,
                ease: "easeInOut",
              }}
            />
            <motion.div
              className={cn(
                "absolute inset-0 border-2 border-primary/30 rounded-full",
                sizeClasses[size]
              )}
              animate={{
                scale: [1, 1.5, 1],
                opacity: [0.5, 0, 0.5],
              }}
              transition={{
                duration: 2,
                repeat: Infinity,
                delay: 0.5,
                ease: "easeInOut",
              }}
            />
          </div>
        );

      default:
        return (
          <motion.div
            className={cn(
              "border-2 border-muted border-t-primary rounded-full",
              sizeClasses[size]
            )}
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
          />
        );
    }
  };

  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center space-y-3",
        className
      )}
    >
      {renderSpinner()}

      {text && (
        <motion.p
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className={cn("text-muted-foreground text-center", textSizes[size])}
        >
          {text}
        </motion.p>
      )}

      {showProgress && (
        <div className="w-full max-w-xs">
          <div className="flex justify-between text-xs text-muted-foreground mb-1">
            <span>Loading...</span>
            <span>{Math.round(progress)}%</span>
          </div>
          <div className="w-full bg-muted rounded-full h-1.5 overflow-hidden">
            <motion.div
              className="h-full bg-primary rounded-full"
              initial={{ width: 0 }}
              animate={{ width: `${progress}%` }}
              transition={{ duration: 0.3, ease: "easeOut" }}
            />
          </div>
        </div>
      )}
    </div>
  );
}
