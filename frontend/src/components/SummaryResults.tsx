import { Shield, AlertTriangle, Skull, Clock, File, HardDrive, Download, BarChart3 } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { ApiPredictResponse, getDownloadUrl } from "@/lib/api";
import { toast } from "@/hooks/use-toast";

interface SummaryResultsProps {
  result: ApiPredictResponse;
  processingTime: number;
}

export const SummaryResults = ({ result, processingTime }: SummaryResultsProps) => {
  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };

  const handleDownload = () => {
    const filename = result.download_csv.split('/').pop() || 'predictions.csv';
    const url = getDownloadUrl(filename);
    window.open(url, '_blank');
    toast({
      title: "Download Started",
      description: "CSV file download initiated",
    });
  };

  const summary = result.summary || {};
  const totalFlows = result.total_flows || 0;
  const benignCount = summary.BENIGN || 0;
  const attackCount = summary.ATTACK || 0;
  const anomalyCount = summary.ANOMALY || 0;

  // Determine primary classification based on counts
  const getPrimaryClassification = (): { 
    type: "benign" | "anomaly" | "attack"; 
    count: number; 
    label: string 
  } => {
    if (attackCount > 0) return { type: 'attack' as const, count: attackCount, label: 'ATTACK' };
    if (anomalyCount > 10) return { type: 'anomaly' as const, count: anomalyCount, label: 'ANOMALY' };
    return { type: 'benign' as const, count: benignCount, label: 'BENIGN' };
  };

  const primary = getPrimaryClassification();

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

  const currentConfig = config[primary.type as keyof typeof config];
  const Icon = currentConfig.icon;

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Main Summary Card */}
      <Card variant={primary.type} className="overflow-hidden">
        <CardHeader className={cn("text-center py-8", currentConfig.bgClass)}>
          <div className="flex flex-col items-center space-y-4">
            <div
              className={cn(
                "p-6 rounded-full",
                currentConfig.bgClass,
                primary.type === "attack" && "animate-pulse-glow"
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
            <p className="text-center text-muted-foreground max-w-md">
              {currentConfig.description}
            </p>
          </div>
        </CardHeader>

        <CardContent className="py-6 space-y-6">
          {/* Statistics Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card className="bg-status-benign/10 border-status-benign/30">
              <CardContent className="p-4 text-center">
                <Shield className="w-8 h-8 text-status-benign mx-auto mb-2" />
                <p className="text-3xl font-bold text-status-benign">{benignCount}</p>
                <p className="text-sm text-muted-foreground">BENIGN</p>
                <p className="text-xs text-muted-foreground mt-1">
                  {totalFlows > 0 ? ((benignCount / totalFlows) * 100).toFixed(1) : 0}%
                </p>
              </CardContent>
            </Card>

            <Card className="bg-status-attack/10 border-status-attack/30">
              <CardContent className="p-4 text-center">
                <Skull className="w-8 h-8 text-status-attack mx-auto mb-2" />
                <p className="text-3xl font-bold text-status-attack">{attackCount}</p>
                <p className="text-sm text-muted-foreground">ATTACK</p>
                <p className="text-xs text-muted-foreground mt-1">
                  {totalFlows > 0 ? ((attackCount / totalFlows) * 100).toFixed(1) : 0}%
                </p>
              </CardContent>
            </Card>

            <Card className="bg-status-anomaly/10 border-status-anomaly/30">
              <CardContent className="p-4 text-center">
                <AlertTriangle className="w-8 h-8 text-status-anomaly mx-auto mb-2" />
                <p className="text-3xl font-bold text-status-anomaly">{anomalyCount}</p>
                <p className="text-sm text-muted-foreground">ANOMALY</p>
                <p className="text-xs text-muted-foreground mt-1">
                  {totalFlows > 0 ? ((anomalyCount / totalFlows) * 100).toFixed(1) : 0}%
                </p>
              </CardContent>
            </Card>
          </div>

          {/* Attack Types Breakdown */}
          {result.attack_types && Object.keys(result.attack_types).length > 0 && (
            <Card className="bg-card/50">
              <CardHeader className="py-4">
                <CardTitle className="text-sm text-muted-foreground font-mono uppercase tracking-wider flex items-center space-x-2">
                  <BarChart3 className="w-4 h-4" />
                  <span>Attack Types Breakdown</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="py-4">
                <div className="flex flex-wrap gap-2">
                  {Object.entries(result.attack_types)
                    .sort((a, b) => b[1] - a[1])
                    .map(([type, count]) => (
                      <div
                        key={type}
                        className="px-4 py-2 bg-status-attack/10 border border-status-attack/30 rounded-lg"
                      >
                        <span className="text-status-attack font-mono text-sm">{type}</span>
                        <span className="text-status-attack/70 ml-2 font-bold">{count}</span>
                      </div>
                    ))}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Download Button */}
          <Button
            variant="outline"
            size="lg"
            className="w-full"
            onClick={handleDownload}
          >
            <Download className="w-4 h-4 mr-2" />
            Download Full Results CSV
          </Button>
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
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="flex items-center space-x-3 p-3 bg-secondary/50 rounded-lg">
              <File className="w-5 h-5 text-primary" />
              <div>
                <p className="text-xs text-muted-foreground">File Name</p>
                <p className="text-sm font-mono truncate max-w-[150px]">
                  {result.filename}
                </p>
              </div>
            </div>

            <div className="flex items-center space-x-3 p-3 bg-secondary/50 rounded-lg">
              <HardDrive className="w-5 h-5 text-primary" />
              <div>
                <p className="text-xs text-muted-foreground">File Size</p>
                <p className="text-sm font-mono">
                  {formatFileSize(result.file_size_bytes)}
                </p>
              </div>
            </div>

            <div className="flex items-center space-x-3 p-3 bg-secondary/50 rounded-lg">
              <BarChart3 className="w-5 h-5 text-primary" />
              <div>
                <p className="text-xs text-muted-foreground">Total Flows</p>
                <p className="text-sm font-mono">{totalFlows}</p>
              </div>
            </div>

            <div className="flex items-center space-x-3 p-3 bg-secondary/50 rounded-lg">
              <Clock className="w-5 h-5 text-primary" />
              <div>
                <p className="text-xs text-muted-foreground">Processing Time</p>
                <p className="text-sm font-mono">
                  {processingTime.toFixed(2)}s
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
