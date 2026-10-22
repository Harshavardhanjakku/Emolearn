import { GamesSection } from "@/components/GamesSection";

const Gaming = () => (
  <section className="space-y-4">
    <div>
      <p className="page-kicker">Practice</p>
      <h2 className="page-title">Games for your learning mood</h2>
      <p className="mt-2 max-w-2xl text-muted-foreground">
        Choose a focused challenge and keep your study session engaging.
      </p>
    </div>
    <GamesSection onClose={() => undefined} />
  </section>
);

export default Gaming;
