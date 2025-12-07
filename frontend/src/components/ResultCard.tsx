import { Shield, AlertTriangle, Skull, Clock, File, HardDrive } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

export interface DetectionResult {
  classification: "benign" | "anomaly" | "attack";
  attack_type?: string;
  confidence: number;
}

interface ProcessingInfo {
  fileName: string;
  fileSize: number;
  processingTime: number;
}

interface ResultCardProps {
  result: DetectionResult;
  processingInfo: ProcessingInfo;
}

export const ResultCard = ({ result, processingInfo }: ResultCardProps) => {
  const config = {
    benign: {
      icon: Shield,
      label: "BENIGN",
      emoji: "✅",
      color: "text-status-benign",
      glowClass: "text-glow-green",
      bgClass: "bg-status-benign/10",
      description: "Network traffic appears normal. No threats detected.",
    },
    anomaly: {
      icon: AlertTriangle,
      label: "ANOMALY",
      emoji: "⚠️",
      color: "text-status-anomaly",
      glowClass: "text-glow-yellow",
      bgClass: "bg-status-anomaly/10",
      description: "Unusual traffic patterns detected. Further investigation recommended.",
    },
    attack: {
      icon: Skull,
      label: "ATTACK",
      emoji: "🚨",
      color: "text-status-attack",
      glowClass: "text-glow-red",
      bgClass: "bg-status-attack/10",
      description: "Malicious traffic detected! Immediate action required.",
    },
  };

  const currentConfig = config[result.classification];
  const Icon = currentConfig.icon;

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Main Result Card */}
      <Card variant={result.classification} className="overflow-hidden">
        <CardHeader className={cn("text-center py-8", currentConfig.bgClass)}>
          <div className="flex flex-col items-center space-y-4">
            <div
              className={cn(
                "p-6 rounded-full",
                currentConfig.bgClass,
                result.classification === "attack" && "animate-pulse-glow"
              )}
            >
              <Icon className={cn("w-16 h-16", currentConfig.color)} />
            </div>
            <CardTitle
              className={cn(
                "text-4xl md:text-5xl tracking-widest",
                currentConfig.color,
                currentConfig.glowClass
              )}
            >
              {currentConfig.label} {currentConfig.emoji}
            </CardTitle>
          </div>
        </CardHeader>

        <CardContent className="py-6 space-y-6">
          <p className="text-center text-muted-foreground">
            {currentConfig.description}
          </p>

          {/* Attack Type (only for attacks) */}
          {result.classification === "attack" && result.attack_type && (
            <div className="text-center p-4 bg-status-attack/10 rounded-lg border border-status-attack/30">
              <p className="text-xs text-muted-foreground uppercase tracking-wider mb-2">
                Attack Type Identified
              </p>
              <p className="text-2xl font-display text-status-attack text-glow-red tracking-wide">
                {result.attack_type}
              </p>
            </div>
          )}

          {/* Confidence Meter */}
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Detection Confidence</span>
              <span className={cn("font-mono font-bold", currentConfig.color)}>
                {(result.confidence * 100).toFixed(1)}%
              </span>
            </div>
            <div className="h-3 bg-secondary rounded-full overflow-hidden">
              <div
                className={cn(
                  "h-full rounded-full transition-all duration-1000 ease-out",
                  result.classification === "benign" && "bg-status-benign",
                  result.classification === "anomaly" && "bg-status-anomaly",
                  result.classification === "attack" && "bg-status-attack"
                )}
                style={{ width: `${result.confidence * 100}%` }}
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Processing Info Card */}
      <Card className="bg-card/50">
        <CardHeader className="py-4">
          <CardTitle className="text-sm text-muted-foreground font-mono uppercase tracking-wider">
            Processing Information
          </CardTitle>
        </CardHeader>
        <CardContent className="py-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="flex items-center space-x-3 p-3 bg-secondary/50 rounded-lg">
              <File className="w-5 h-5 text-primary" />
              <div>
                <p className="text-xs text-muted-foreground">File Name</p>
                <p className="text-sm font-mono truncate max-w-[150px]">
                  {processingInfo.fileName}
                </p>
              </div>
            </div>

            <div className="flex items-center space-x-3 p-3 bg-secondary/50 rounded-lg">
              <HardDrive className="w-5 h-5 text-primary" />
              <div>
                <p className="text-xs text-muted-foreground">File Size</p>
                <p className="text-sm font-mono">
                  {formatFileSize(processingInfo.fileSize)}
                </p>
              </div>
            </div>

            <div className="flex items-center space-x-3 p-3 bg-secondary/50 rounded-lg">
              <Clock className="w-5 h-5 text-primary" />
              <div>
                <p className="text-xs text-muted-foreground">Processing Time</p>
                <p className="text-sm font-mono">
                  {processingInfo.processingTime.toFixed(2)}s
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
