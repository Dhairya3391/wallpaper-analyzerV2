import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { motion } from "framer-motion";

import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium ring-offset-background transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 relative overflow-hidden",
  {
    variants: {
      variant: {
        default:
          "bg-primary text-primary-foreground hover:bg-primary/90 active:scale-95",
        destructive:
          "bg-destructive text-destructive-foreground hover:bg-destructive/90 active:scale-95",
        outline:
          "border border-input bg-background hover:bg-accent hover:text-accent-foreground active:scale-95",
        secondary:
          "bg-secondary text-secondary-foreground hover:bg-secondary/80 active:scale-95",
        ghost: "hover:bg-accent hover:text-accent-foreground active:scale-95",
        link: "text-primary underline-offset-4 hover:underline active:scale-95",
        glass:
          "glass border border-border/50 hover:border-border active:scale-95",
        gradient:
          "bg-gradient-to-r from-primary to-primary/80 text-primary-foreground hover:from-primary/90 hover:to-primary/70 active:scale-95",
      },
      size: {
        default: "h-10 px-4 py-2",
        sm: "h-9 rounded-md px-3",
        lg: "h-11 rounded-md px-8",
        xl: "h-14 rounded-lg px-10 text-base",
        icon: "h-10 w-10",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
  loading?: boolean;
  loadingText?: string;
  ripple?: boolean;
  rippleColor?: string;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      className,
      variant,
      size,
      asChild = false,
      loading = false,
      loadingText,
      ripple = true,
      rippleColor = "rgba(255, 255, 255, 0.3)",
      children,
      disabled,
      ...props
    },
    ref
  ) => {
    const [ripples, setRipples] = React.useState<
      Array<{ id: number; x: number; y: number }>
    >([]);
    const rippleId = React.useRef(0);

    const handleRipple = (event: React.MouseEvent<HTMLButtonElement>) => {
      if (!ripple || loading || disabled) return;

      const button = event.currentTarget;
      const rect = button.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;

      const newRipple = {
        id: rippleId.current++,
        x,
        y,
      };

      setRipples(prev => [...prev, newRipple]);

      // Remove ripple after animation
      setTimeout(() => {
        setRipples(prev => prev.filter(r => r.id !== newRipple.id));
      }, 600);
    };

    const Comp = asChild ? Slot : "button";

    return (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        disabled={disabled || loading}
        onClick={handleRipple}
        {...props}
      >
        {/* Ripple Effects */}
        {ripples.map(ripple => (
          <motion.span
            key={ripple.id}
            className="absolute rounded-full pointer-events-none"
            style={{
              left: ripple.x,
              top: ripple.y,
              backgroundColor: rippleColor,
              width: 0,
              height: 0,
            }}
            initial={{ width: 0, height: 0, x: -0, y: -0 }}
            animate={{
              width: 300,
              height: 300,
              x: -150,
              y: -150,
              opacity: [1, 0],
            }}
            transition={{
              duration: 0.6,
              ease: "easeOut",
            }}
          />
        ))}

        {/* Loading State */}
        {loading ? (
          <div className="flex items-center gap-2">
            <motion.div
              className="w-4 h-4 border-2 border-current border-t-transparent rounded-full"
              animate={{ rotate: 360 }}
              transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
            />
            {loadingText || children}
          </div>
        ) : (
          children
        )}
      </Comp>
    );
  }
);
Button.displayName = "Button";

export { Button, buttonVariants };
