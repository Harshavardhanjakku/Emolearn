import { AchievementGrid, StatsGrid, WeeklyGoalCard } from "@/components/ProgressBadges";

const Achievements = () => (
  <div className="space-y-6">
    <div>
      <p className="page-kicker">Progress</p>
      <h2 className="page-title">Achievements</h2>
      <p className="mt-2 max-w-2xl text-muted-foreground">
        Track your consistency, celebrate milestones, and keep building momentum.
      </p>
    </div>
    <StatsGrid />
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_20rem]">
      <AchievementGrid />
      <WeeklyGoalCard />
    </div>
  </div>
);

export default Achievements;
