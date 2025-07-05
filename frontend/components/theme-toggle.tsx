"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { 
  Check, 
  Sun, 
  Moon, 
  Monitor,
  Waves,
  Sunset,
  Trees,
  Sparkles,
  Star
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  DropdownMenuSeparator,
  DropdownMenuLabel,
} from "@/components/ui/dropdown-menu";
import { useTheme } from "@/components/theme-provider";

const themes = [
  {
    name: "light",
    label: "Light",
    icon: Sun,
    gradient: "from-yellow-400 to-orange-500",
    description: "Clean and bright",
  },
  {
    name: "dark",
    label: "Dark",
    icon: Moon,
    gradient: "from-gray-700 to-gray-900",
    description: "Easy on the eyes",
  },
  {
    name: "ocean",
    label: "Ocean",
    icon: Waves,
    gradient: "from-blue-400 to-cyan-500",
    description: "Deep blue vibes",
  },
  {
    name: "sunset",
    label: "Sunset",
    icon: Sunset,
    gradient: "from-orange-400 to-pink-500",
    description: "Warm and cozy",
  },
  {
    name: "forest",
    label: "Forest",
    icon: Trees,
    gradient: "from-green-600 to-emerald-700",
    description: "Natural and fresh",
  },
  {
    name: "purple",
    label: "Purple",
    icon: Sparkles,
    gradient: "from-purple-600 to-pink-600",
    description: "Royal and elegant",
  },
  {
    name: "midnight",
    label: "Midnight",
    icon: Star,
    gradient: "from-indigo-900 to-purple-900",
    description: "Deep and mysterious",
  },
  {
    name: "system",
    label: "System",
    icon: Monitor,
    gradient: "from-gray-500 to-gray-700",
    description: "Follow system preference",
  },
];

export function ThemeToggle() {
  const { theme, setTheme } = useTheme();
  const [isOpen, setIsOpen] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const currentTheme = themes.find((t) => t.name === theme) || themes[0];

  if (!mounted) {
    return (
      <Button
        variant="outline"
        size="sm"
        className="gap-2 bg-background/50 backdrop-blur-sm border-border/50"
      >
        <div className="w-4 h-4 rounded-full bg-gradient-to-r from-gray-400 to-gray-600" />
        <span className="hidden sm:inline">Theme</span>
      </Button>
    );
  }

  return (
    <DropdownMenu open={isOpen} onOpenChange={setIsOpen}>
      <DropdownMenuTrigger asChild>
        <Button
          variant="outline"
          size="sm"
          className="gap-2 bg-background/50 backdrop-blur-sm border-border/50 hover:bg-background/80 transition-all duration-200"
        >
          <motion.div
            key={currentTheme.name}
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.2 }}
            className={`w-4 h-4 rounded-full bg-gradient-to-r ${currentTheme.gradient} shadow-sm`}
          />
          <span className="hidden sm:inline font-medium">{currentTheme.label}</span>
        </Button>
      </DropdownMenuTrigger>
      
      <DropdownMenuContent 
        align="end" 
        className="w-64 p-2 bg-background/95 backdrop-blur-md border-border/50"
        sideOffset={8}
      >
        <DropdownMenuLabel className="px-2 py-1.5">
          <div className="flex flex-col">
            <span className="text-sm font-semibold">Choose Theme</span>
            <span className="text-xs text-muted-foreground">
              Customize your experience
            </span>
          </div>
        </DropdownMenuLabel>
        
        <DropdownMenuSeparator className="my-2" />
        
        <div className="space-y-1">
          <AnimatePresence>
            {themes.map((themeOption, index) => (
              <motion.div
                key={themeOption.name}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.2, delay: index * 0.03 }}
              >
                <DropdownMenuItem
                  onClick={() => setTheme(themeOption.name as any)}
                  className="flex items-center justify-between cursor-pointer p-3 rounded-lg hover:bg-muted/50 transition-all duration-200 group"
                >
                  <div className="flex items-center gap-3">
                    <div className="relative">
                      <div
                        className={`w-6 h-6 rounded-full bg-gradient-to-r ${themeOption.gradient} shadow-sm ring-2 ring-transparent group-hover:ring-border transition-all duration-200`}
                      />
                      <themeOption.icon className="absolute inset-0 w-3 h-3 m-auto text-white drop-shadow-sm" />
                    </div>
                    <div className="flex flex-col">
                      <span className="text-sm font-medium">{themeOption.label}</span>
                      <span className="text-xs text-muted-foreground">
                        {themeOption.description}
                      </span>
                    </div>
                  </div>
                  
                  {theme === themeOption.name && (
                    <motion.div
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      transition={{ duration: 0.2 }}
                      className="flex items-center justify-center"
                    >
                      <Check className="h-4 w-4 text-primary" />
                    </motion.div>
                  )}
                </DropdownMenuItem>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
        
        <DropdownMenuSeparator className="my-2" />
        
        <div className="px-2 py-1">
          <p className="text-xs text-muted-foreground">
            Themes automatically adapt to your content
          </p>
        </div>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}