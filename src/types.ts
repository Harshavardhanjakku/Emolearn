export type Explanation = {
  title: string;
  overview: string;
  keyConcepts: string[];
  examples: Array<{
    title: string;
    code: string;
  }>;
  learningPath: string[];
};

