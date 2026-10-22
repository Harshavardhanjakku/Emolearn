import { Menu, PanelLeftClose, PanelLeftOpen } from "lucide-react";
import { useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { AIAssistant } from "@/components/AIAssistant";
import BreathingExercise from "@/components/BreathingExercise";
import { QuizDialog } from "@/components/QuizDialog";
import { ThemeToggle } from "@/components/theme/ThemeToggle";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { useSession } from "@/contexts/session-context";
import { emotionToEmoji } from "@/lib/emotions";
import { cn } from "@/lib/utils";
import { NAV_ITEMS } from "./nav";

const pageMeta: Record<string, { kicker: string; title: string }> = {
  "/": { kicker: "Workspace", title: "Overview" },
  "/dashboard": { kicker: "Workspace", title: "Overview" },
  "/learning": { kicker: "Curriculum", title: "Learning" },
  "/gaming": { kicker: "Practice", title: "Gaming" },
  "/achievements": { kicker: "Progress", title: "Achievements" },
  "/insights": { kicker: "Analytics", title: "Emotion Insights" },
  "/focus": { kicker: "Session", title: "Focus & Study" },
  "/break": { kicker: "Recharge", title: "Take a Break" },
};

const NavList = ({ onNavigate, collapsed = false }: { onNavigate?: () => void; collapsed?: boolean }) => (
  <nav className="flex flex-col gap-0.5" aria-label="Primary">
    {NAV_ITEMS.map((item) => {
      const Icon = item.icon;
      return (
        <NavLink
          key={item.to}
          to={item.to}
          onClick={onNavigate}
          title={collapsed ? item.label : undefined}
          className={({ isActive }) =>
            cn(
              "flex items-center gap-2.5 rounded-md px-2.5 py-2 text-sm transition-colors",
              collapsed && "justify-center px-2",
              isActive
                ? "bg-sidebar-accent text-sidebar-accent-foreground"
                : "text-sidebar-foreground/75 hover:bg-sidebar-accent/70 hover:text-sidebar-foreground"
            )
          }
        >
          {({ isActive }) => (
            <>
              <Icon className={cn("h-4 w-4 shrink-0", isActive ? "text-primary" : "text-muted-foreground")} aria-hidden />
              <span className={cn("min-w-0 overflow-hidden whitespace-nowrap transition-all duration-300", collapsed ? "w-0 opacity-0" : "w-auto opacity-100")}>
                <span className="block truncate font-medium leading-5">{item.label}</span>
              </span>
            </>
          )}
        </NavLink>
      );
    })}
  </nav>
);

export const AppShell = () => {
  const location = useLocation();
  const {
    currentEmotion,
    isAssistantOpen,
    isEmotionTriggered,
    isBreathingOpen,
    setIsBreathingOpen,
    handleAssistantClose,
    isDnd,
    detectionStatus,
  } = useSession();
  const meta = pageMeta[location.pathname] ?? { kicker: "EmoLearn", title: "Studio" };
  const [mobileOpen, setMobileOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="flex min-h-screen">
        <aside
          className={cn(
            "sticky top-0 hidden h-screen shrink-0 flex-col border-r border-sidebar-border bg-sidebar px-3 py-4 transition-[width] duration-300 ease-in-out md:flex",
            sidebarCollapsed ? "w-[4.5rem]" : "w-64"
          )}
        >
          <div className={cn("mb-5 flex items-center gap-2.5 px-1", sidebarCollapsed && "justify-center")}>
            <div className="flex h-9 w-9 items-center justify-center rounded-md bg-primary text-primary-foreground">
              <span className="font-display text-lg font-extrabold">E</span>
            </div>
            <div className={cn("min-w-0 overflow-hidden transition-all duration-300", sidebarCollapsed ? "w-0 opacity-0" : "w-auto opacity-100")}>
              <p className="font-display text-lg font-bold leading-none">EmoLearn</p>
              <p className="mt-1 text-xs text-muted-foreground">Adaptive study studio</p>
            </div>
          </div>
          <NavList collapsed={sidebarCollapsed} />
        </aside>

        <div className="flex min-w-0 flex-1 flex-col">
          <header className="sticky top-0 z-30 flex items-center justify-between gap-3 border-b-2 border-border bg-background/90 px-4 py-3 backdrop-blur md:px-6">
            <div className="flex min-w-0 items-center gap-3">
              <Sheet open={mobileOpen} onOpenChange={setMobileOpen}>
                <SheetTrigger asChild>
                  <Button variant="outline" size="icon" className="md:hidden" aria-label="Open navigation">
                    <Menu className="h-4 w-4" />
                  </Button>
                </SheetTrigger>
                <SheetContent side="left" className="w-72 bg-sidebar p-4">
                  <SheetHeader>
                    <SheetTitle className="text-left font-display">EmoLearn</SheetTitle>
                  </SheetHeader>
                  <div className="mt-4">
                    <NavList onNavigate={() => setMobileOpen(false)} />
                  </div>
                </SheetContent>
              </Sheet>
              <Button
                type="button"
                variant="outline"
                size="icon"
                className="hidden shrink-0 md:inline-flex"
                onClick={() => setSidebarCollapsed((collapsed) => !collapsed)}
                aria-label={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
                aria-pressed={sidebarCollapsed}
              >
                {sidebarCollapsed ? <PanelLeftOpen className="h-4 w-4" /> : <PanelLeftClose className="h-4 w-4" />}
              </Button>
              <div className="min-w-0">
                <p className="page-kicker">{meta.kicker}</p>
                <h1 className="truncate font-display text-lg font-bold sm:text-xl">{meta.title}</h1>
              </div>
            </div>

            <div className="flex items-center gap-2">
              {isDnd && (
                <span className="hidden border-2 border-warning/40 bg-warning/15 px-2 py-1 text-xs font-semibold text-warning sm:inline">
                  Do not disturb
                </span>
              )}
              <div className="surface-inset hidden items-center gap-2 rounded-md border border-border px-2.5 py-1.5 sm:flex">
                <span aria-hidden>{emotionToEmoji(currentEmotion)}</span>
                <span className="text-xs font-medium">
                  {currentEmotion ?? (detectionStatus === "idle" ? "No detection" : "Waiting")}
                </span>
              </div>
              <ThemeToggle />
            </div>
          </header>

          <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-6 md:px-6">
            <Outlet />
          </main>
        </div>
      </div>

      <AIAssistant
        isOpen={isAssistantOpen}
        onClose={handleAssistantClose}
        currentEmotion={emotionToEmoji(currentEmotion)}
        isEmotionTriggered={isEmotionTriggered}
      />
      <BreathingExercise isOpen={isBreathingOpen} onClose={() => setIsBreathingOpen(false)} />
      <QuizDialog />
    </div>
  );
};
