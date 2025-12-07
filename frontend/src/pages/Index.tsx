import { useState, useEffect } from "react";
import { Crosshair, RotateCcw } from "lucide-react";
import { CyberBackground } from "@/components/CyberBackground";
import { Header } from "@/components/Header";
import { FileUpload } from "@/components/FileUpload";
import { LoadingAnimation } from "@/components/LoadingAnimation";
import { SummaryResults } from "@/components/SummaryResults";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { toast } from "@/hooks/use-toast";
import { predictFile, checkApiHealth, ApiPredictResponse } from "@/lib/api";

const Index = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<ApiPredictResponse | null>(null);
  const [processingTime, setProcessingTime] = useState<number>(0);
  const [apiConnected, setApiConnected] = useState<boolean | null>(null);

  // Check API health on mount
  useEffect(() => {
    checkApiHealth()
      .then(() => {
        setApiConnected(true);
      })
      .catch(() => {
        setApiConnected(false);
        toast({
          title: "API Connection Failed",
          description: "Cannot connect to API server. Please ensure the backend is running on port 8000.",
          variant: "destructive",
        });
      });
  }, []);

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    setResult(null);
    setProcessingTime(0);
  };

  const handleRunDetection = async () => {
    if (!selectedFile) {
      toast({
        title: "No file selected",
        description: "Please upload a .pcap or .csv file first.",
        variant: "destructive",
      });
      return;
    }

    setIsAnalyzing(true);
    const startTime = performance.now();

    try {
      const apiResult = await predictFile(selectedFile);

      const endTime = performance.now();
      const processingTime = (endTime - startTime) / 1000;

      setResult(apiResult);
      setProcessingTime(processingTime);

      // Show success toast with summary
      const summary = apiResult.summary || {};
      const attackCount = summary.ATTACK || 0;
      const anomalyCount = summary.ANOMALY || 0;
      const benignCount = summary.BENIGN || 0;

      if (attackCount > 0) {
        toast({
          title: "Analysis Complete - Threats Detected!",
          description: `Found ${attackCount} attack(s), ${anomalyCount} anomaly(ies), and ${benignCount} benign flow(s).`,
          variant: "destructive",
        });
      } else if (anomalyCount > 0) {
        toast({
          title: "Analysis Complete - Anomalies Detected",
          description: `Found ${anomalyCount} anomaly(ies) and ${benignCount} benign flow(s).`,
        });
      } else {
        toast({
          title: "Analysis Complete",
          description: `All ${benignCount} flow(s) appear to be benign.`,
        });
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "An error occurred while processing the file.";
      toast({
        title: "Analysis Failed",
        description: errorMessage,
        variant: "destructive",
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setResult(null);
    setProcessingTime(0);
  };

  return (
    <div className="min-h-screen bg-background text-foreground relative overflow-hidden">
      <CyberBackground />

      <div className="relative z-10 flex flex-col min-h-screen">
        <Header />

        <main className="flex-1 container mx-auto px-4 py-8 md:py-12">
          <div className="max-w-3xl mx-auto space-y-8">
            {/* Title Section */}
            <div className="text-center space-y-4 animate-fade-in">
              <h2 className="text-3xl md:text-4xl font-display font-bold tracking-wide">
                Network Traffic <span className="text-primary text-glow-cyan">Analysis</span>
              </h2>
              <p className="text-muted-foreground max-w-xl mx-auto">
                Upload network capture files for real-time threat detection powered by
                advanced machine learning algorithms.
              </p>
            </div>

            {/* Upload Section */}
            {!isAnalyzing && !result && (
              <Card className="animate-fade-in" style={{ animationDelay: "0.1s" }}>
                <CardHeader>
                  <CardTitle className="text-lg font-display flex items-center space-x-2">
                    <Crosshair className="w-5 h-5 text-primary" />
                    <span>Upload Network Capture</span>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-6">
                  <FileUpload
                    onFileSelect={handleFileSelect}
                    selectedFile={selectedFile}
                    disabled={isAnalyzing}
                  />

                  {apiConnected === false && (
                    <div className="p-3 bg-status-attack/10 border border-status-attack/30 rounded-lg text-center">
                      <p className="text-sm text-status-attack">
                        ⚠️ API server not connected. Please start the backend server.
                      </p>
                    </div>
                  )}
                  <Button
                    variant="cyber"
                    size="xl"
                    className="w-full"
                    onClick={handleRunDetection}
                    disabled={!selectedFile || isAnalyzing || apiConnected === false}
                  >
                    <Crosshair className="w-5 h-5" />
                    Run Detection
                  </Button>
                </CardContent>
              </Card>
            )}

            {/* Loading State */}
            {isAnalyzing && (
              <Card className="animate-fade-in">
                <CardContent className="py-4">
                  <LoadingAnimation />
                </CardContent>
              </Card>
            )}

            {/* Results */}
            {result && (
              <div className="space-y-6">
                <SummaryResults result={result} processingTime={processingTime} />

                <div className="flex justify-center">
                  <Button variant="outline" size="lg" onClick={handleReset}>
                    <RotateCcw className="w-4 h-4" />
                    Analyze Another File
                  </Button>
                </div>
              </div>
            )}
          </div>
        </main>

        {/* Footer */}
        <footer className="relative z-10 border-t border-border/50 bg-background/80 backdrop-blur-sm py-4">
          <div className="container mx-auto px-4 text-center">
            <p className="text-xs text-muted-foreground font-mono">
              AI-IDS v1.0 • Powered by Machine Learning • © 2024
            </p>
          </div>
        </footer>
      </div>
    </div>
  );
};

export default Index;
