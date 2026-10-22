export type Lesson = {
  id: string;
  title: string;
  chapter: string;
  course: string;
  durationLabel: string;
};

export const COURSE_NAME = "Algorithms fundamentals";

export const lessons: Lesson[] = [
  {
    id: "0IAPZzGSbME",
    title: "Introduction to Algorithms",
    chapter: "Introduction to Algorithms",
    course: COURSE_NAME,
    durationLabel: "Lesson 1",
  },
  {
    id: "9TlHvipP5yA",
    title: "Time Complexity (Part 1)",
    chapter: "Time Complexity (Part 1)",
    course: COURSE_NAME,
    durationLabel: "Lesson 2",
  },
  {
    id: "gBz44smaa9A",
    title: "Time Complexities of Searching & Sorting",
    chapter: "Time Complexities of Searching & Sorting",
    course: COURSE_NAME,
    durationLabel: "Lesson 3",
  },
];
