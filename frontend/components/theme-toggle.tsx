"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Sun, Moon, Monitor } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useTheme } from "@/components/theme-provider";

const themes = [
  {
    name: "light",
    label: "Light",
    icon: Sun,
    description: "Clean and bright",
  },
  {
    name: "dark",
    label: "Dark",
    icon: Moon,
    description: "Easy on the eyes",
  },
  {
    name: "system",
    label: "System",
    icon: Monitor,
    description: "Follow system preference",
  },
];

export function ThemeToggle() {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const currentTheme = themes.find(t => t.name === theme) || themes[0];

  if (!mounted) {
    return (
      <Button
        variant="outline"
        size="sm"
        className="gap-2 glass border-border/50"
      >
        <div className="w-4 h-4 rounded-full bg-muted" />
        <span className="hidden sm:inline">Theme</span>
      </Button>
    );
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="outline"
          size="sm"
          className="gap-2 glass border-border/50 hover:bg-muted/50 transition-all duration-200"
        >
          <motion.div
            key={currentTheme.name}
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.2 }}
          >
            <currentTheme.icon className="w-4 h-4" />
          </motion.div>
          <span className="hidden sm:inline font-medium">
            {currentTheme.label}
          </span>
        </Button>
      </DropdownMenuTrigger>

      <DropdownMenuContent
        align="end"
        className="w-48 glass border-border/50"
        sideOffset={8}
      >
        {themes.map(themeOption => (
          <DropdownMenuItem
            key={themeOption.name}
            onClick={() =>
              setTheme(themeOption.name as "light" | "dark" | "system")
            }
            className="flex items-center gap-3 cursor-pointer p-3 rounded-lg hover:bg-muted/50 transition-all duration-200"
          >
            <themeOption.icon className="w-4 h-4" />
            <div className="flex flex-col">
              <span className="text-sm font-medium">{themeOption.label}</span>
              <span className="text-xs text-muted-foreground">
                {themeOption.description}
              </span>
            </div>
            {theme === themeOption.name && (
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                className="ml-auto w-2 h-2 rounded-full bg-primary"
              />
            )}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
