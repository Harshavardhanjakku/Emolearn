import { useState } from "react";
import { Coffee } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import BreathingExercise from "@/components/BreathingExercise";

const Break = () => {
  const [isBreakOpen, setIsBreakOpen] = useState(false);

  return (
    <>
      <div className="mx-auto max-w-2xl space-y-6">
        <div>
          <p className="page-kicker">Recharge</p>
          <h2 className="page-title">Take a Break</h2>
          <p className="mt-2 text-muted-foreground">
            Reset your attention with music, motivation, a timer, and a quick reflex activity.
          </p>
        </div>
        <Card className="p-6">
          <div className="flex flex-col items-start gap-4 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <h3 className="font-display text-lg font-bold">Brain Reset</h3>
              <p className="mt-1 text-sm text-muted-foreground">
                Start a two-minute guided break whenever you need to recharge.
              </p>
            </div>
            <Button onClick={() => setIsBreakOpen(true)}>
              <Coffee className="h-4 w-4" />
              Start break
            </Button>
          </div>
        </Card>
      </div>
      <BreathingExercise isOpen={isBreakOpen} onClose={() => setIsBreakOpen(false)} />
    </>
  );
};

export default Break;
