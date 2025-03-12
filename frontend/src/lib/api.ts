// TODO: move to environment variable
const API_BASE_URL = "http://localhost:8000"; // Backend FastAPI server URL

export interface NGram {
    0: string; // n-gram text
    1: number; // count
}

export interface NamedEntity {
  0: string; // entity text
  1: number; // count
  2: number; // sentiment score from [-1, 1]
  3: string; // summarized sentiment
}

export interface SubredditQuery {
    name: string;
    time_filter: string;
    sort_by: string;
}

export interface SubredditAnalysis {
    top_n_grams: Record<string, NGram[]>;
    top_named_entities: Record<string, NamedEntity[]>;
}

export interface SubredditAnalysisResponse {
    analysis_status: string;
    analysis_progress: number;
    analysis: SubredditAnalysis;
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
): Promise<SubredditAnalysisResponse> {
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
