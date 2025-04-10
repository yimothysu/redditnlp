const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export interface NGram {
    0: string; // n-gram text
    1: number; // count
}

export interface NamedEntity {
    0: string; // entity text
    1: number; // count
    2: number; // sentiment score from [-1, 1]
    3: string; // summarized sentiment
    4: string[]; // links 
}

export interface SubredditQuery {
    name: string;
    time_filter: string;
    sort_by: string;
}

export interface SubredditAnalysis {
    timestamp: number;
    num_words: number;
    subreddit: string;

    top_n_grams: Record<string, NGram[]>;
    top_named_entities: Record<string, NamedEntity[]>;
    top_named_entities_embeddings: Record<string, [number, number]>;
    top_named_entities_wordcloud: string;
    readability_metrics: Record<string, number>;

    // For subreddit ranking
    toxicity_score: number; // [0, 1] --> 0 = not toxic at all, 1 = all toxic
    toxicity_grade: string; // A+ to F
    toxicity_percentile: number; // [0, 100]
    all_toxicity_scores: [number]; // for generating a toxicity scores distribution graph on the UI
    all_toxicity_grades: [string]; // for generating a toxicity grades distribution graph on the UI

    positive_content_score: number; // [0, 1] --> 0 = no positive content, 1 = all positive content
    positive_content_grade: string; // A+ to F
    positive_content_percentile: number; // [0, 100]
    all_positive_content_scores: [number]; // for generating a positive content scores distribution graph on the UI
    all_positive_content_grades: [string]; // for generating a positive content grades distribution graph on the UI
}

export interface Subreddit {
    name: string;
    description: string;
}

export interface PopularSubredditsResponse {
    reddits: Subreddit[];
}

export interface SubredditRequest {
    subreddit: string;
    email: string;
}

// TODO: Update doc
/**
 * Fetches NLP analysis data for a subreddit
 * @param subreddit The name of the subreddit to analyze
 * @param timeFilter Time period filter (all, year, month, week)
 * @param sortBy How to sort posts (top, controversial)
 * @returns Promise with the subreddit NLP analysis
 */
export async function fetchSubredditAnalysis(
    subredditQuery: SubredditQuery
): Promise<SubredditAnalysis> {
    const response = await fetch(`${API_BASE_URL}/analysis/`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(subredditQuery),
    });

    if (!response.ok) {
        throw new Error(
            `Failed to fetch subreddit analysis: ${response.statusText}`
        );
    }

    return await response.json();
}

export async function fetchPopularSubreddits(): Promise<PopularSubredditsResponse> {
    const response = await fetch(`${API_BASE_URL}/popular_subreddits/`, {
        method: "GET",
        headers: {
            "Content-Type": "application/json",
        },
    });

    if (!response.ok) {
        throw new Error(
            `Failed to fetch popular subreddits: ${response.statusText}`
        );
    }

    return await response.json();
}

export async function requestSubreddit(subreddit: string, email: string) {
    return fetch(`${API_BASE_URL}/request_subreddit/`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            subreddit: subreddit,
            email: email,
        }),
    });
}
