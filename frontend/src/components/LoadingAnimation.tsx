import { cn } from "@/lib/utils";

export const LoadingAnimation = () => {
  return (
    <div className="flex flex-col items-center justify-center py-12 space-y-8">
      {/* Radar/Scanner animation */}
      <div className="relative w-40 h-40">
        {/* Outer ring */}
        <div className="absolute inset-0 rounded-full border-2 border-primary/30" />
        <div className="absolute inset-2 rounded-full border border-primary/20" />
        <div className="absolute inset-4 rounded-full border border-primary/10" />

        {/* Center dot */}
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="w-3 h-3 rounded-full bg-primary glow-cyan animate-pulse" />
        </div>

        {/* Scanning beam */}
        <div
          className="absolute inset-0 origin-center animate-radar-spin"
          style={{
            background: `conic-gradient(from 0deg, transparent 0deg, hsl(180 100% 50% / 0.3) 30deg, transparent 60deg)`,
          }}
        />

        {/* Blip points */}
        <div className="absolute top-6 left-1/2 -translate-x-1/2 w-2 h-2 rounded-full bg-primary/60 animate-pulse" />
        <div className="absolute bottom-10 right-8 w-1.5 h-1.5 rounded-full bg-primary/40 animate-pulse" style={{ animationDelay: "0.5s" }} />
        <div className="absolute top-1/2 left-8 w-1.5 h-1.5 rounded-full bg-primary/50 animate-pulse" style={{ animationDelay: "1s" }} />
      </div>

      {/* Status text */}
      <div className="text-center space-y-2">
        <h3 className="text-xl font-display text-primary text-glow-cyan tracking-wider">
          ANALYZING NETWORK TRAFFIC
        </h3>
        <div className="flex items-center justify-center space-x-2">
          <span className="text-muted-foreground font-mono text-sm">Processing packets</span>
          <span className="flex space-x-1">
            {[0, 1, 2].map((i) => (
              <span
                key={i}
                className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse"
                style={{ animationDelay: `${i * 0.3}s` }}
              />
            ))}
          </span>
        </div>
      </div>

      {/* Progress bar */}
      <div className="w-64 h-1 bg-secondary rounded-full overflow-hidden">
        <div
          className="h-full bg-gradient-to-r from-primary via-accent to-primary animate-data-stream"
          style={{
            width: "100%",
            backgroundSize: "200% 100%",
            animation: "data-stream 1.5s linear infinite",
          }}
        />
      </div>

      {/* Metrics */}
      <div className="flex space-x-8 text-xs font-mono text-muted-foreground">
        <div className="flex items-center space-x-2">
          <div className="w-2 h-2 rounded-full bg-status-benign animate-pulse" />
          <span>Parsing headers</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-2 h-2 rounded-full bg-status-anomaly animate-pulse" style={{ animationDelay: "0.3s" }} />
          <span>Feature extraction</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-2 h-2 rounded-full bg-primary animate-pulse" style={{ animationDelay: "0.6s" }} />
          <span>ML inference</span>
        </div>
      </div>
    </div>
  );
};
