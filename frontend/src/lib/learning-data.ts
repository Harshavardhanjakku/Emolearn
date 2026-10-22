import { Brain, Clock, Heart, Star, Target, Trophy, type LucideIcon } from "lucide-react";

export type LearningStat = {
  id: string;
  label: string;
  value: string;
  hint: string;
};

export const learningStats: LearningStat[] = [
  { id: "streak", label: "Study streak", value: "7 days", hint: "Consecutive days with a session" },
  { id: "focus", label: "Focus score", value: "85%", hint: "From completed focus sessions" },
  { id: "questions", label: "Questions asked", value: "23", hint: "Across AI help and quizzes" },
  { id: "concepts", label: "Concepts mastered", value: "12", hint: "Marked complete in lessons" },
];

export type Achievement = {
  id: string;
  icon: LucideIcon;
  title: string;
  description: string;
  earned: boolean;
  progressLabel?: string;
  progress?: number;
};

export const achievements: Achievement[] = [
  {
    id: "focus-champ",
    icon: Trophy,
    title: "Focus Champ",
    description: "1 hour without distractions",
    earned: true,
  },
  {
    id: "calm-master",
    icon: Heart,
    title: "Calm Master",
    description: "Recovered from frustration",
    earned: true,
  },
  {
    id: "curious-learner",
    icon: Brain,
    title: "Curious Learner",
    description: "Asked 10+ questions",
    earned: false,
    progress: 100,
    progressLabel: "23 / 10 questions logged",
  },
  {
    id: "goal-achiever",
    icon: Target,
    title: "Goal Achiever",
    description: "Completed daily target",
    earned: false,
    progress: 60,
    progressLabel: "12 / 20 weekly hours",
  },
  {
    id: "time-master",
    icon: Clock,
    title: "Time Master",
    description: "5 focused sessions",
    earned: true,
  },
  {
    id: "quiz-master",
    icon: Star,
    title: "Quiz Master",
    description: "90% quiz accuracy",
    earned: false,
  },
];

export const weeklyGoal = {
  label: "Study hours",
  current: 12,
  target: 20,
  unit: "hrs",
};

export const weeklyGoalPercent = Math.round((weeklyGoal.current / weeklyGoal.target) * 100);
export const weeklyGoalRemaining = Math.max(weeklyGoal.target - weeklyGoal.current, 0);
