import { Shield, Activity } from "lucide-react";

export const Header = () => {
  return (
    <header className="relative z-10 border-b border-border/50 bg-background/80 backdrop-blur-sm">
      <div className="container mx-auto px-4 py-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="relative">
              <Shield className="w-10 h-10 text-primary" />
              <div className="absolute -top-1 -right-1 w-3 h-3 rounded-full bg-status-benign animate-pulse" />
            </div>
            <div>
              <h1 className="text-2xl md:text-3xl font-display font-bold text-foreground tracking-wider">
                AI-<span className="text-primary text-glow-cyan">IDS</span>
              </h1>
              <p className="text-xs text-muted-foreground font-mono uppercase tracking-widest">
                Intrusion Detection System
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-4">
            <div className="hidden md:flex items-center space-x-2 px-4 py-2 bg-secondary/50 rounded-lg border border-border/50">
              <Activity className="w-4 h-4 text-status-benign" />
              <span className="text-xs font-mono text-muted-foreground">
                System Status:
              </span>
              <span className="text-xs font-mono text-status-benign">ONLINE</span>
            </div>
            <div className="flex items-center space-x-1">
              {[0, 1, 2].map((i) => (
                <div
                  key={i}
                  className="w-2 h-6 bg-primary/30 rounded-sm animate-pulse"
                  style={{
                    animationDelay: `${i * 0.2}s`,
                    height: `${12 + i * 6}px`,
                  }}
                />
              ))}
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};
