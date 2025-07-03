"use client";

import { Settings } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
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

interface AnalysisSettings {
  directory: string;
  similarity_threshold: number;
  aesthetic_threshold: number;
  recursive: boolean;
  skip_duplicates: boolean;
  skip_aesthetics: boolean;
  limit: number;
}

interface SettingsModalProps {
  settings: AnalysisSettings;
  onSettingsChange: (settings: AnalysisSettings) => void;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function SettingsModal({
  settings,
  onSettingsChange,
  open,
  onOpenChange,
}: SettingsModalProps) {
  const updateSetting = <K extends keyof AnalysisSettings>(
    key: K,
    value: AnalysisSettings[K],
  ) => {
    onSettingsChange({ ...settings, [key]: value });
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm">
          <Settings className="w-4 h-4 mr-2" />
          Settings
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Analysis Settings</DialogTitle>
        </DialogHeader>

        <div className="space-y-6">
          <div className="space-y-2">
            <Label htmlFor="directory">Directory Path</Label>
            <Input
              id="directory"
              value={settings.directory}
              onChange={(e) => updateSetting("directory", e.target.value)}
              placeholder="Enter directory path..."
            />
          </div>

          <div className="space-y-2">
            <Label>
              Similarity Threshold: {settings.similarity_threshold.toFixed(2)}
            </Label>
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
          </div>

          <div className="space-y-2">
            <Label>
              Aesthetic Threshold: {settings.aesthetic_threshold.toFixed(2)}
            </Label>
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
          </div>

          <div className="space-y-2">
            <Label htmlFor="limit">Image Limit</Label>
            <Select
              value={settings.limit.toString()}
              onValueChange={(value) =>
                updateSetting("limit", Number.parseInt(value))
              }
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="0">No Limit</SelectItem>
                <SelectItem value="100">100 images</SelectItem>
                <SelectItem value="500">500 images</SelectItem>
                <SelectItem value="1000">1000 images</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <Label htmlFor="recursive">Include Subdirectories</Label>
              <Switch
                id="recursive"
                checked={settings.recursive}
                onCheckedChange={(checked) =>
                  updateSetting("recursive", checked)
                }
              />
            </div>

            <div className="flex items-center justify-between">
              <Label htmlFor="skip_duplicates">Skip Duplicate Detection</Label>
              <Switch
                id="skip_duplicates"
                checked={settings.skip_duplicates}
                onCheckedChange={(checked) =>
                  updateSetting("skip_duplicates", checked)
                }
              />
            </div>

            <div className="flex items-center justify-between">
              <Label htmlFor="skip_aesthetics">Skip Aesthetic Analysis</Label>
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
      </DialogContent>
    </Dialog>
  );
}
