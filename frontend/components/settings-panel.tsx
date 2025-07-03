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
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";

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
  const updateSetting = <K extends keyof AnalysisSettings>(
    key: K,
    value: AnalysisSettings[K],
  ) => {
    onSettingsChange({ ...settings, [key]: value });
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-xl">
            <Settings2 className="w-6 h-6" />
            Analysis Settings
          </DialogTitle>
        </DialogHeader>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          className="space-y-8 py-4"
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
                  onChange={(e) => updateSetting("directory", e.target.value)}
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
                  onCheckedChange={(checked) =>
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
                  onCheckedChange={(checked) =>
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
                  onCheckedChange={(checked) =>
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
              <Select
                value={settings.limit.toString()}
                onValueChange={(value) =>
                  updateSetting("limit", Number.parseInt(value))
                }
              >
                <SelectTrigger className="mt-1">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="0">No Limit</SelectItem>
                  <SelectItem value="100">100 images</SelectItem>
                  <SelectItem value="500">500 images</SelectItem>
                  <SelectItem value="1000">1,000 images</SelectItem>
                  <SelectItem value="5000">5,000 images</SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground mt-1">
                Limit the number of images to process for faster analysis
              </p>
            </div>
          </div>
        </motion.div>
      </DialogContent>
    </Dialog>
  );
}
