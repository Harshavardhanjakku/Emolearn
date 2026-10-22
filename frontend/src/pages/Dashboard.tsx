import { ControlPanel } from "@/components/ControlPanel";
import { EmotionTracker } from "@/components/EmotionTracker";
import { MoodGraph } from "@/components/MoodGraph";
import { VideoPlayer } from "@/components/VideoPlayer";

const Dashboard = () => (
  <div className="min-w-0 space-y-6">
    <VideoPlayer />
    <ControlPanel compact />
    <EmotionTracker />
    <MoodGraph />
  </div>
);

export default Dashboard;
