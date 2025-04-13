export type NGram = [string, number];

export type NamedEntity = [
  string,
  number,
  number,
  string,
  Record<string, string>
];

export interface SubredditAnalysis {
  timestamp: number;
  num_words: number;
  toxicity_score: number;
  toxicity_grade: string;
  toxicity_percentile: number;
  all_toxicity_scores: number[];
  all_toxicity_grades: string[];
  positive_content_score: number;
  positive_content_grade: string;
  positive_content_percentile: number;
  all_positive_content_scores: number[];
  all_positive_content_grades: string[];
  readability_metrics: {
    avg_num_words_title: number;
    avg_num_words_description: number;
    avg_flesch_grade_level: number;
    dale_chall_grade_levels: Record<string, number>;
  };
  top_named_entities: Record<string, NamedEntity[]>;
  top_n_grams: Record<string, NGram[]>;
  top_named_entities_wordcloud: string;
  top_named_entities_embeddings: Record<string, [number, number]>;
}

export interface RedditAnalysisDisplayProps {
  name?: string;
  inComparisonMode: string;
  currentMenuItem?: string;
  setCurrentMenuItem?: (item: string) => void;
}
