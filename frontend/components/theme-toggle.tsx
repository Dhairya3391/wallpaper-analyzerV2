"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Check, Sun, Moon, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useTheme } from "next-themes";

const themes = [
  {
    name: "light",
    label: "Light",
    icon: Sun,
    gradient: "from-yellow-400 to-orange-500",
  },
  {
    name: "dark",
    label: "Dark",
    icon: Moon,
    gradient: "from-gray-700 to-gray-900",
  },
  {
    name: "valentine",
    label: "Valentine",
    icon: Sparkles,
    gradient: "from-pink-400 to-red-500",
  },
  {
    name: "synthwave",
    label: "Synthwave",
    icon: Sparkles,
    gradient: "from-purple-600 to-pink-600",
  },
  {
    name: "cyberpunk",
    label: "Cyberpunk",
    icon: Sparkles,
    gradient: "from-cyan-400 to-purple-600",
  },
  {
    name: "forest",
    label: "Forest",
    icon: Sparkles,
    gradient: "from-green-600 to-emerald-700",
  },
  {
    name: "luxury",
    label: "Luxury",
    icon: Sparkles,
    gradient: "from-amber-500 to-yellow-600",
  },
  {
    name: "dracula",
    label: "Dracula",
    icon: Sparkles,
    gradient: "from-purple-800 to-indigo-900",
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
    // Optionally, render a placeholder or nothing
    return (
      <Button
        variant="outline"
        size="sm"
        className="gap-2 bg-background/50 backdrop-blur-sm"
      />
    );
  }

  return (
    <DropdownMenu open={isOpen} onOpenChange={setIsOpen}>
      <DropdownMenuTrigger asChild>
        <Button
          variant="outline"
          size="sm"
          className="gap-2 bg-background/50 backdrop-blur-sm"
        >
          <div
            className={`w-4 h-4 rounded-full bg-gradient-to-r ${currentTheme.gradient}`}
          />
          <span className="hidden sm:inline">Theme</span>
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-56 p-2">
        <div className="mb-2 px-2 py-1">
          <p className="text-sm font-medium">Choose Theme</p>
          <p className="text-xs text-muted-foreground">
            Customize your experience
          </p>
        </div>
        <AnimatePresence>
          {themes.map((themeOption, index) => (
            <motion.div
              key={themeOption.name}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.2, delay: index * 0.03 }}
            >
              <DropdownMenuItem
                onClick={() => setTheme(themeOption.name)}
                className="flex items-center justify-between cursor-pointer p-3 rounded-lg hover:bg-muted/50"
              >
                <div className="flex items-center gap-3">
                  <div
                    className={`w-5 h-5 rounded-full bg-gradient-to-r ${themeOption.gradient} shadow-sm`}
                  />
                  <div>
                    <p className="text-sm font-medium">{themeOption.label}</p>
                  </div>
                </div>
                {theme === themeOption.name && (
                  <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ duration: 0.2 }}
                  >
                    <Check className="h-4 w-4 text-primary" />
                  </motion.div>
                )}
              </DropdownMenuItem>
            </motion.div>
          ))}
        </AnimatePresence>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
