"use client";

import { motion } from "framer-motion";
import { Settings, Github, Heart, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ThemeToggle } from "@/components/theme-toggle";

interface HeaderProps {
  onSettingsClick: () => void;
}

export function Header({ onSettingsClick }: HeaderProps) {
  return (
    <motion.header
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
      className="sticky top-0 z-50 w-full border-b border-border/50 bg-background/80 backdrop-blur-md supports-[backdrop-filter]:bg-background/60"
    >
      <div className="container mx-auto px-4">
        <div className="flex h-16 items-center justify-between">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="flex items-center space-x-3"
          >
            <div className="relative group">
              <div className="w-10 h-10 gradient-primary rounded-xl flex items-center justify-center shadow-lg group-hover:shadow-glow transition-all duration-300">
                <span className="text-white font-bold text-lg">W</span>
              </div>
              <motion.div 
                className="absolute -top-1 -right-1 w-4 h-4 bg-gradient-to-br from-green-400 to-blue-500 rounded-full"
                animate={{ scale: [1, 1.2, 1] }}
                transition={{ duration: 2, repeat: Infinity }}
              />
            </div>
            <div>
              <h1 className="text-xl font-bold gradient-text">
                Wallyzer
              </h1>
              <p className="text-xs text-muted-foreground">AI Image Curator</p>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="flex items-center space-x-2"
          >
            <Button 
              variant="ghost" 
              size="sm" 
              className="hidden sm:flex hover:bg-muted/50 transition-all duration-200"
            >
              <Github className="w-4 h-4 mr-2" />
              GitHub
            </Button>
            <Button 
              variant="ghost" 
              size="sm" 
              className="hidden sm:flex hover:bg-muted/50 transition-all duration-200 group"
            >
              <Heart className="w-4 h-4 mr-2 group-hover:text-red-500 transition-colors duration-200" />
              Sponsor
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={onSettingsClick}
              className="hover:bg-muted/50 transition-all duration-200"
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