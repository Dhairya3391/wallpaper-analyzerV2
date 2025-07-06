"use client";

import { motion } from "framer-motion";
import { Settings, Github } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ThemeToggle } from "@/components/theme-toggle";
import { useIsMobile } from "@/hooks/use-mobile";

interface HeaderProps {
  onSettingsClick: () => void;
}

export function Header({ onSettingsClick }: HeaderProps) {
  const isMobile = useIsMobile();

  return (
    <motion.header
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
      className="sticky top-0 z-50 w-full border-b border-border/50 glass supports-[backdrop-filter]:bg-background/60"
    >
      <div className={`container mx-auto ${isMobile ? "px-4" : "px-6"}`}>
        <div className="flex h-16 items-center justify-between">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="flex items-center space-x-4"
          >
            <div className="relative group">
              <div className="w-10 h-10 gradient-primary rounded-xl flex items-center justify-center shadow-medium group-hover:shadow-strong transition-all duration-300">
                <span className="text-primary-foreground font-bold text-lg">
                  W
                </span>
              </div>
              <motion.div
                className="absolute -top-1 -right-1 w-3 h-3 bg-accent rounded-full"
                animate={{ scale: [1, 1.2, 1] }}
                transition={{ duration: 2, repeat: Infinity }}
              />
            </div>
            <div>
              <h1
                className={`${isMobile ? "text-lg" : "text-xl"} font-bold text-foreground`}
              >
                Wallyzer
              </h1>
              <p className="text-xs text-muted-foreground">AI Image Curator</p>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="flex items-center space-x-3"
          >
            {!isMobile && (
              <a
                href="https://github.com/Dhairya3391/wallyzer"
                target="_blank"
                rel="noopener noreferrer"
              >
                <Button variant="ghost" size="sm" className="btn-ghost">
                  <Github className="w-4 h-4 mr-2" />
                  GitHub
                </Button>
              </a>
            )}
            <Button
              variant="ghost"
              size="sm"
              onClick={onSettingsClick}
              className="btn-ghost"
            >
              <Settings className="w-4 h-4" />
            </Button>
            <ThemeToggle />
          </motion.div>
        </div>
      </div>
    </motion.header>
  );
}
