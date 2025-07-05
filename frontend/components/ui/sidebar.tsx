"use client";

import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { ChevronLeft, ChevronRight } from "lucide-react";

import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

const sidebarVariants = cva(
  "fixed top-0 left-0 h-full z-50 bg-background border-r transition-all duration-300 ease-in-out",
  {
    variants: {
      state: {
        open: "w-72 p-4",
        closed: "w-16 p-2",
      },
    },
    defaultVariants: {
      state: "open",
    },
  }
);

export interface SidebarProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof sidebarVariants> {
  children: React.ReactNode;
}

const Sidebar = React.forwardRef<HTMLDivElement, SidebarProps>(
  ({ className, state, children, ...props }, ref) => {
    const [isOpen, setIsOpen] = React.useState(state === "open");

    const toggleSidebar = () => {
      setIsOpen(!isOpen);
    };

    return (
      <div
        ref={ref}
        className={cn(
          sidebarVariants({ state: isOpen ? "open" : "closed" }),
          className
        )}
        {...props}
      >
        <div className="flex justify-end mb-4">
          <Button variant="ghost" size="icon" onClick={toggleSidebar}>
            {isOpen ? (
              <ChevronLeft className="h-4 w-4" />
            ) : (
              <ChevronRight className="h-4 w-4" />
            )}
          </Button>
        </div>
        <div
          className={cn(
            "transition-opacity duration-300",
            isOpen ? "opacity-100" : "opacity-0"
          )}
        >
          {isOpen && children}
        </div>
      </div>
    );
  }
);
Sidebar.displayName = "Sidebar";

export { Sidebar };
