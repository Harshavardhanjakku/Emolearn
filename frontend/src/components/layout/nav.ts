import {
  BookOpen,
  Coffee,
  Crosshair,
  Gamepad2,
  LayoutDashboard,
  Sparkles,
  Trophy,
  type LucideIcon,
} from "lucide-react";

export type NavItem = {
  to: string;
  label: string;
  icon: LucideIcon;
  description: string;
};

export const NAV_ITEMS: NavItem[] = [
  { to: "/dashboard", label: "Overview", icon: LayoutDashboard, description: "Streak, goals, and session" },
  { to: "/learning", label: "Learning", icon: BookOpen, description: "Courses and lesson player" },
  { to: "/gaming", label: "Gaming", icon: Gamepad2, description: "Educational challenges" },
  { to: "/achievements", label: "Achievements", icon: Trophy, description: "Badges and milestones" },
  { to: "/insights", label: "Emotion Insights", icon: Sparkles, description: "Mood and detection" },
  { to: "/focus", label: "Focus & Study", icon: Crosshair, description: "Timer and session controls" },
  { to: "/break", label: "Take a Break", icon: Coffee, description: "Reset, breathe, and recharge" },
];
