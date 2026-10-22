import { ControlPanel } from "@/components/ControlPanel";
import { WeeklyGoalCard } from "@/components/ProgressBadges";

const Focus = () => (
  <div className="space-y-6">
    <div>
      <p className="page-kicker">Session</p>
      <h2 className="page-title">Focus & Study</h2>
      <p className="mt-2 max-w-2xl text-muted-foreground">
        Set up your sensors, quiz yourself, and protect a distraction-free study session.
      </p>
    </div>
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_20rem]">
      <ControlPanel />
      <WeeklyGoalCard />
    </div>
  </div>
);

export default Focus;
