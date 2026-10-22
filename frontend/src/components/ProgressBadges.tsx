import { Link } from "react-router-dom";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { achievements, learningStats, weeklyGoal, weeklyGoalPercent, weeklyGoalRemaining } from "@/lib/learning-data";
import { cn } from "@/lib/utils";

export const StatsGrid = ({ compact = false }: { compact?: boolean }) => {
  const items = compact ? learningStats.slice(0, 4) : learningStats;
  return (
    <div className={cn("grid gap-3", compact ? "grid-cols-2" : "grid-cols-2 lg:grid-cols-4")}>
      {items.map((stat, index) => (
        <Card key={stat.id} className={cn("p-4", index === 0 && "border-primary/40")}>
          <p className="text-xs font-medium text-muted-foreground">{stat.label}</p>
          <p className="mt-1 font-display text-2xl font-bold">{stat.value}</p>
          {!compact && <p className="mt-1 text-xs text-muted-foreground">{stat.hint}</p>}
        </Card>
      ))}
    </div>
  );
};

export const WeeklyGoalCard = () => (
  <Card className="p-5">
    <div className="mb-3 flex items-start justify-between gap-3">
      <div>
        <p className="page-kicker">This week</p>
        <h3 className="font-display text-lg font-bold">Weekly goal</h3>
      </div>
      <span className="text-sm font-semibold">
        {weeklyGoal.current} / {weeklyGoal.target} {weeklyGoal.unit}
      </span>
    </div>
    <Progress value={weeklyGoalPercent} aria-label="Weekly study hours progress" className="h-2.5 border border-border" />
    <p className="mt-2 text-sm text-muted-foreground">{weeklyGoalRemaining} hours remaining this week.</p>
    <Button asChild className="mt-4" size="sm">
      <Link to="/focus">Continue a focus session</Link>
    </Button>
  </Card>
);

export const AchievementGrid = ({ limit }: { limit?: number }) => {
  const items = limit ? achievements.slice(0, limit) : achievements;
  return (
    <div className="grid gap-3 sm:grid-cols-2">
      {items.map((badge) => {
        const Icon = badge.icon;
        const state = badge.earned ? "earned" : badge.progress ? "in-progress" : "locked";
        return (
          <Card
            key={badge.id}
            className={cn(
              "p-4",
              state === "earned" && "border-primary/45",
              state === "locked" && "opacity-75"
            )}
          >
            <div className="flex items-start gap-3">
              <div
                className={cn(
                  "flex h-10 w-10 items-center justify-center rounded-md border-2",
                  state === "earned" ? "border-primary bg-primary/15" : "border-border bg-muted"
                )}
              >
                <Icon className="h-5 w-5" />
              </div>
              <div className="min-w-0 flex-1">
                <div className="flex items-center justify-between gap-2">
                  <h4 className="font-display text-sm font-semibold">{badge.title}</h4>
                  <Badge variant={state === "earned" ? "default" : "outline"}>
                    {state === "earned" ? "Earned" : state === "in-progress" ? "In progress" : "Locked"}
                  </Badge>
                </div>
                <p className="mt-1 text-xs text-muted-foreground">{badge.description}</p>
                {state === "in-progress" && badge.progress != null && (
                  <div className="mt-2">
                    <Progress value={badge.progress} className="h-1.5" />
                    <p className="mt-1 text-[11px] text-muted-foreground">{badge.progressLabel}</p>
                  </div>
                )}
              </div>
            </div>
          </Card>
        );
      })}
    </div>
  );
};

export const ProgressBadges = () => (
  <div className="space-y-4">
    <Card className="p-5">
      <h3 className="mb-3 font-display text-lg font-bold">Learning stats</h3>
      <div className="space-y-3">
        {learningStats.map((stat) => (
          <div key={stat.id} className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">{stat.label}</span>
            <span className="font-semibold">{stat.value}</span>
          </div>
        ))}
      </div>
    </Card>
    <AchievementGrid limit={4} />
    <WeeklyGoalCard />
  </div>
);
