import { useCallback, useState } from "react";
import { Upload, FileWarning, CheckCircle2 } from "lucide-react";
import { cn } from "@/lib/utils";

interface FileUploadProps {
  onFileSelect: (file: File) => void;
  selectedFile: File | null;
  disabled?: boolean;
}

export const FileUpload = ({ onFileSelect, selectedFile, disabled }: FileUploadProps) => {
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const validateFile = (file: File): boolean => {
    const fileName = file.name.toLowerCase();
    if (!fileName.endsWith(".pcap") && !fileName.endsWith(".csv")) {
      setError("Invalid file type. Only .pcap and .csv files are accepted.");
      return false;
    }
    setError(null);
    return true;
  };

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setIsDragging(false);

      if (disabled) return;

      const file = e.dataTransfer.files[0];
      if (file && validateFile(file)) {
        onFileSelect(file);
      }
    },
    [onFileSelect, disabled]
  );

  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file && validateFile(file)) {
        onFileSelect(file);
      }
    },
    [onFileSelect]
  );

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };

  return (
    <div className="w-full">
      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        className={cn(
          "relative border-2 border-dashed rounded-lg p-8 transition-all duration-300 cursor-pointer group",
          isDragging && !disabled
            ? "border-primary bg-primary/10 glow-cyan"
            : selectedFile
            ? "border-status-benign bg-status-benign/5 border-glow-green"
            : "border-border hover:border-primary/50 hover:bg-primary/5",
          disabled && "opacity-50 cursor-not-allowed",
          error && "border-status-attack bg-status-attack/5"
        )}
      >
        <input
          type="file"
          accept=".pcap,.csv"
          onChange={handleFileInput}
          disabled={disabled}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer disabled:cursor-not-allowed"
        />

        <div className="flex flex-col items-center justify-center space-y-4 text-center">
          {error ? (
            <>
              <div className="p-4 rounded-full bg-status-attack/20 text-status-attack">
                <FileWarning className="w-10 h-10" />
              </div>
              <div>
                <p className="text-status-attack font-medium">{error}</p>
                <p className="text-sm text-muted-foreground mt-1">
                  Please select a valid .pcap or .csv file
                </p>
              </div>
            </>
          ) : selectedFile ? (
            <>
              <div className="p-4 rounded-full bg-status-benign/20 text-status-benign animate-scale-in">
                <CheckCircle2 className="w-10 h-10" />
              </div>
              <div className="animate-fade-in">
                <p className="text-status-benign font-medium font-display">
                  FILE LOADED
                </p>
                <p className="text-sm text-foreground mt-2 font-mono">
                  {selectedFile.name}
                </p>
                <p className="text-xs text-muted-foreground mt-1">
                  {formatFileSize(selectedFile.size)}
                </p>
              </div>
            </>
          ) : (
            <>
              <div
                className={cn(
                  "p-4 rounded-full bg-primary/10 text-primary transition-all duration-300",
                  "group-hover:bg-primary/20 group-hover:scale-110"
                )}
              >
                <Upload className="w-10 h-10" />
              </div>
              <div>
                <p className="text-foreground font-display tracking-wide">
                  DROP FILE HERE
                </p>
                <p className="text-sm text-muted-foreground mt-2">
                  or click to browse
                </p>
                <p className="text-xs text-primary/70 mt-4 font-mono">
                  Accepts: .pcap and .csv files
                </p>
              </div>
            </>
          )}
        </div>

        {/* Scan line effect */}
        {isDragging && !disabled && (
          <div className="absolute inset-0 overflow-hidden rounded-lg pointer-events-none">
            <div className="absolute inset-x-0 h-1/3 scan-line" />
          </div>
        )}
      </div>
    </div>
  );
};
