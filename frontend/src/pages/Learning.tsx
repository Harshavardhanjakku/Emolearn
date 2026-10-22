import { AITutorSection } from "@/components/AITutorSection";
import { ControlPanel } from "@/components/ControlPanel";
import { EmotionTracker } from "@/components/EmotionTracker";
import { VideoPlayer } from "@/components/VideoPlayer";

const Learning = () => (
  <div className="space-y-6">
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_20rem]">
      <div className="min-w-0 space-y-6">
        <VideoPlayer />
        <ControlPanel />
      </div>
      <EmotionTracker />
    </div>
    <AITutorSection onClose={() => undefined} />
  </div>
);

export default Learning;
