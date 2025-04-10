import { useState, useEffect, useRef } from "react";
import { fetchSubredditAnalysis } from "../lib/api";
import { SubredditAvatar } from "./SubredditAvatar";
import { RedditAnalysisDisplayProps } from "../types/redditAnalysis";
import { AnalysisMenu } from "./analysis/AnalysisMenu";
import { ReadabilityMetrics } from "./analysis/ReadabilityMetrics";
import { WordCloud } from "./analysis/WordCloud";
import { WordEmbeddings } from "./analysis/WordEmbeddings";
import { NGrams } from "./analysis/NGrams";
import { ComparativeAnalysis } from "./analysis/ComparativeAnalysis";
import { NamedEntities } from "./analysis/NamedEntities";

export default function RedditAnalysisDisplay({
  name,
  inComparisonMode,
}: RedditAnalysisDisplayProps) {
  const [error, setError] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<any | null>(null);
  const [currentMenuItem, setCurrentMenuItem] = useState("Named Entities");
  const [isBusy, setIsBusy] = useState(false);
  const [timeFilter, setTimeFilter] = useState("week");
  const [sortBy, setSortBy] = useState("top");

  const isMounted = useRef(true);

  useEffect(() => {
    setAnalysis(null); // reset when timeFilter changes
    loadSubredditAnalysis();
  }, [timeFilter]);

  useEffect(() => {
    isMounted.current = true;
    return () => {
      isMounted.current = false;
    };
  }, []);

  const loadSubredditAnalysis = async () => {
    if (isBusy || !name || name.trim().length === 0) return;
    setIsBusy(true);
    try {
      const analysis = await fetchSubredditAnalysis({
        name: name.trim(),
        time_filter: timeFilter,
        sort_by: sortBy,
      });
      setError(null);
      setAnalysis(analysis);
    } catch (error) {
      setError(
        error instanceof Error ? error.message : "An unknown error occurred"
      );
    } finally {
      setIsBusy(false);
    }
  };

  useEffect(() => {
    loadSubredditAnalysis();
  }, []);

  const DisplayedAnalysis = () => {
    if (!analysis) return null;

    switch (currentMenuItem) {
      case "Ranking":
        return timeFilter === "all" ? (
          <ComparativeAnalysis analysis={analysis} />
        ) : null;
      case "Bi-Grams":
        return <NGrams analysis={analysis} timeFilter={timeFilter} />;
      case "Named Entities":
        return <NamedEntities analysis={analysis} timeFilter={timeFilter} />;
      case "Word Embeddings":
        return (
          <WordEmbeddings
            analysis={analysis}
            inComparisonMode={inComparisonMode}
          />
        );
      case "Word Cloud":
        return <WordCloud analysis={analysis} />;
      case "Readability Metrics":
        return <ReadabilityMetrics analysis={analysis} />;
      default:
        return null;
    }
  };

  const spinnerStyle =
    "animate-spin h-6 w-6 border-t-4 border-blue-500 border-solid rounded-full";

  return (
    <div>
      <div className="ml-2 mr-2 mt-4 mb-4 p-5 bg-white rounded-md shadow-sm">
        <div className="flex flex-col items-center mb-6">
          <SubredditAvatar subredditName={name ?? ""} />
          <h1 className="text-lg font-bold">r/{name}</h1>
        </div>
        <div className="flex justify-center items-center gap-4 mb-2">
          <div>
            <label className="block text-sm text-center font-medium text-gray-700">
              Time Filter
            </label>
            <select
              value={timeFilter}
              onChange={(e) => setTimeFilter(e.target.value)}
              className="mt-1 block w-full px-1 py-2 text-base border border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
            >
              <option value="all">All Time</option>
              <option value="year">Year</option>
              <option value="week">Week</option>
            </select>
          </div>
          <div>
            <label className="block text-sm text-center font-medium text-gray-700">
              Sort By
            </label>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="mt-1 block w-full px-1 py-2 text-base border border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
            >
              <option value="top">Top Posts</option>
            </select>
          </div>
          <div className="relative flex gap-4">
            {isBusy && !analysis && (
              <div className={`absolute ${spinnerStyle} right-[-40px]`}></div>
            )}
          </div>
        </div>

        {error && (
          <div className="bg-red-50 border-l-4 border-red-400 p-4 mt-4">
            <div className="flex">
              <div className="ml-3">
                <p className="text-sm text-red-700">Error: {error}</p>
              </div>
            </div>
          </div>
        )}

        {!error && analysis && (
          <div className="mt-4">
            <hr className="my-4 border-t border-gray-400 mx-auto w-[97%]" />
            <div className="flex items-center justify-center">
              <div className="flex flex-col m-2 text-[15px] items-center justify-center text-center">
                <div className="flex gap-2 justify-center items-center mb-4">
                  <h1 className="text-white bg-[#fa6f4d] font-bold w-7 h-7 rounded-full text-center text-[18px]">
                    !
                  </h1>
                  <h1 className="text-[#fa6f4d] font-semibold text-center">
                    This analysis may be outdated
                  </h1>
                </div>
                <h1 className="mb-4">
                  analysis generated on
                  <span className="font-bold p-1 rounded-sm">
                    {new Date(analysis.timestamp * 1000).toLocaleString()}
                  </span>
                </h1>
                <h1>
                  analysis analyzed
                  <span className="font-bold p-1 rounded-sm">
                    {analysis.num_words}
                  </span>
                  words
                </h1>
              </div>
            </div>
            <hr className="my-4 border-t border-gray-400 mx-auto w-[97%]" />
            <div className="flex flex-col">
              <AnalysisMenu
                timeFilter={timeFilter}
                currentMenuItem={currentMenuItem}
                setCurrentMenuItem={setCurrentMenuItem}
              />
              <DisplayedAnalysis />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
