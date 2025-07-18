"use client";

import { motion } from "framer-motion";
import {
  Settings2,
  Folder,
  Sliders,
  ToggleRightIcon as Toggle,
  Hash,
} from "lucide-react";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Separator } from "@/components/ui/separator";
import { useIsMobile } from "@/hooks/use-mobile";

interface AnalysisSettings {
  directory: string;
  similarity_threshold: number;
  aesthetic_threshold: number;
  recursive: boolean;
  skip_duplicates: boolean;
  skip_aesthetics: boolean;
  limit: number;
}

interface SettingsPanelProps {
  settings: AnalysisSettings;
  onSettingsChange: (settings: AnalysisSettings) => void;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function SettingsPanel({
  settings,
  onSettingsChange,
  open,
  onOpenChange,
}: SettingsPanelProps) {
  const isMobile = useIsMobile();

  const updateSetting = <K extends keyof AnalysisSettings>(
    key: K,
    value: AnalysisSettings[K]
  ) => {
    onSettingsChange({ ...settings, [key]: value });
  };

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent
        side="right"
        className={`${isMobile ? "w-full" : "w-[400px]"} overflow-y-auto`}
      >
        <SheetHeader className="pb-6">
          <SheetTitle className="flex items-center gap-2 text-xl">
            <Settings2 className="w-6 h-6" />
            Analysis Settings
          </SheetTitle>
        </SheetHeader>

        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.3 }}
          className="space-y-8"
        >
          {/* Directory Settings */}
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <Folder className="w-5 h-5 text-primary" />
              <h3 className="text-lg font-semibold">Directory Configuration</h3>
            </div>
            <div className="space-y-3">
              <div>
                <Label htmlFor="directory" className="text-sm font-medium">
                  Target Directory
                </Label>
                <Input
                  id="directory"
                  value={settings.directory}
                  onChange={e => updateSetting("directory", e.target.value)}
                  placeholder="Enter the path to your image directory..."
                  className="mt-1"
                />
              </div>
              <div className="flex items-center justify-between">
                <div>
                  <Label htmlFor="recursive" className="text-sm font-medium">
                    Include Subdirectories
                  </Label>
                  <p className="text-xs text-muted-foreground">
                    Scan all subdirectories recursively
                  </p>
                </div>
                <Switch
                  id="recursive"
                  checked={settings.recursive}
                  onCheckedChange={checked =>
                    updateSetting("recursive", checked)
                  }
                />
              </div>
            </div>
          </div>

          <Separator />

          {/* Analysis Thresholds */}
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <Sliders className="w-5 h-5 text-primary" />
              <h3 className="text-lg font-semibold">Analysis Thresholds</h3>
            </div>
            <div className="space-y-6">
              <div>
                <div className="flex justify-between items-center mb-2">
                  <Label className="text-sm font-medium">
                    Similarity Threshold
                  </Label>
                  <span className="text-sm text-muted-foreground">
                    {settings.similarity_threshold.toFixed(2)}
                  </span>
                </div>
                <Slider
                  value={[settings.similarity_threshold]}
                  onValueChange={([value]) =>
                    updateSetting("similarity_threshold", value)
                  }
                  min={0}
                  max={1}
                  step={0.01}
                  className="w-full"
                />
                <p className="text-xs text-muted-foreground mt-1">
                  Higher values detect more similar images for clustering
                </p>
              </div>

              <div>
                <div className="flex justify-between items-center mb-2">
                  <Label className="text-sm font-medium">
                    Aesthetic Threshold
                  </Label>
                  <span className="text-sm text-muted-foreground">
                    {settings.aesthetic_threshold.toFixed(2)}
                  </span>
                </div>
                <Slider
                  value={[settings.aesthetic_threshold]}
                  onValueChange={([value]) =>
                    updateSetting("aesthetic_threshold", value)
                  }
                  min={0}
                  max={1}
                  step={0.01}
                  className="w-full"
                />
                <p className="text-xs text-muted-foreground mt-1">
                  Minimum aesthetic score for image quality filtering
                </p>
              </div>
            </div>
          </div>

          <Separator />

          {/* Processing Options */}
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <Toggle className="w-5 h-5 text-primary" />
              <h3 className="text-lg font-semibold">Processing Options</h3>
            </div>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <Label
                    htmlFor="skip_duplicates"
                    className="text-sm font-medium"
                  >
                    Skip Duplicate Detection
                  </Label>
                  <p className="text-xs text-muted-foreground">
                    Disable duplicate image detection
                  </p>
                </div>
                <Switch
                  id="skip_duplicates"
                  checked={settings.skip_duplicates}
                  onCheckedChange={checked =>
                    updateSetting("skip_duplicates", checked)
                  }
                />
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <Label
                    htmlFor="skip_aesthetics"
                    className="text-sm font-medium"
                  >
                    Skip Aesthetic Analysis
                  </Label>
                  <p className="text-xs text-muted-foreground">
                    Disable AI-powered aesthetic scoring
                  </p>
                </div>
                <Switch
                  id="skip_aesthetics"
                  checked={settings.skip_aesthetics}
                  onCheckedChange={checked =>
                    updateSetting("skip_aesthetics", checked)
                  }
                />
              </div>
            </div>
          </div>

          <Separator />

          {/* Performance Settings */}
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <Hash className="w-5 h-5 text-primary" />
              <h3 className="text-lg font-semibold">Performance</h3>
            </div>
            <div>
              <Label htmlFor="limit" className="text-sm font-medium">
                Processing Limit
              </Label>
              <Input
                id="limit"
                type="number"
                value={settings.limit}
                onChange={e =>
                  updateSetting("limit", Number.parseInt(e.target.value) || 0)
                }
                placeholder="0 for no limit"
                className="mt-1"
              />
              <p className="text-xs text-muted-foreground mt-1">
                Limit the number of images to process for faster analysis (0 =
                no limit)
              </p>
            </div>
          </div>
        </motion.div>
      </SheetContent>
    </Sheet>
  );
}
