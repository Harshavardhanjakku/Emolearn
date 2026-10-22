import { EmotionTracker } from "@/components/EmotionTracker";
import { MoodGraph } from "@/components/MoodGraph";

const Insights = () => (
  <div className="space-y-6">
    <div>
      <p className="page-kicker">Analytics</p>
      <h2 className="page-title">Emotion Insights</h2>
      <p className="mt-2 max-w-2xl text-muted-foreground">
        Review your live emotional state and recent detection trends.
      </p>
    </div>
    <EmotionTracker />
    <MoodGraph />
  </div>
);

export default Insights;
