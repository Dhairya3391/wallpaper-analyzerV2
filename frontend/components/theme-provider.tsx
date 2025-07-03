"use client";

import { ThemeProvider as NextThemesProvider } from "next-themes";
import type { ThemeProviderProps } from "next-themes";

export function ThemeProvider({ children, ...props }: ThemeProviderProps) {
  return (
    <NextThemesProvider
      attribute="class"
      defaultTheme="valentine"
      themes={[
        "light",
        "dark",
        "valentine",
        "synthwave",
        "cyberpunk",
        "forest",
        "luxury",
        "dracula",
      ]}
      enableSystem={false}
      {...props}
    >
      {children}
    </NextThemesProvider>
  );
}
