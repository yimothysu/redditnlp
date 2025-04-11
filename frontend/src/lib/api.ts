import { SubredditAnalysis } from "../types/redditAnalysis";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export interface SubredditQuery {
  name: string;
  time_filter: string;
  sort_by: string;
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

/**
 * Fetches NLP analysis data for a subreddit
 * @param subredditQuery The subreddit query object containing subreddit name, time filter, and sort by
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
