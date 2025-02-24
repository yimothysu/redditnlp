import { useState } from "react";
import { useParams } from "react-router-dom";
import Template from "./Template.tsx";
import { fetchSubredditAnalysis, SubredditNLPAnalysis } from "../src/lib/api";
import { SubredditAvatar } from "../src/components/SubredditAvatar.tsx";

export default function RedditPage() {
  const { name } = useParams();
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<SubredditNLPAnalysis | null>(null);
  const [timeFilter, setTimeFilter] = useState("week");
  const [sortBy, setSortBy] = useState("top");

  const renderNGrams = () => {
    if (!analysis) return null;

    return (
      <div className="mt-4">
        <h2 className="text-xl font-bold mb-2">Top N-Grams</h2>
        {Object.entries(analysis.top_n_grams).map(([date, ngrams]) => (
          <div key={date} className="mb-4">
            <h3 className="text-lg font-semibold">{date}</h3>
            <ul className="list-disc pl-5">
              {ngrams.map((ngram, index) => (
                <li key={index}>
                  {ngram[0]}: {ngram[1]}
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    );
  };

  const renderNamedEntities = () => {
    if (!analysis) return null;

    return (
      <div className="mt-4">
        <h2 className="text-xl font-bold mb-2">Top Named Entities</h2>
        {Object.entries(analysis.top_named_entities).map(([date, entities]) => (
          <div key={date} className="mb-4">
            <h3 className="text-lg font-semibold">{date}</h3>
            <ul className="list-disc pl-5">
              {entities.map((entity, index) => (
                <li key={index}>
                  {entity[0]}: {entity[1]}
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    );
  };

  return (
    <Template>
      <div className="m-4 p-4 bg-white rounded-md">
        <div className="flex flex-col items-center mb-6">
          <SubredditAvatar subredditName={name ?? ""} />
          <h1 className="text-lg font-bold">r/{name}</h1>
        </div>

        <div className="mb-4">
          <div className="flex gap-4 mb-2">
            <div>
              <label className="block text-sm font-medium text-gray-700">Time Filter</label>
              <select
                value={timeFilter}
                onChange={(e) => setTimeFilter(e.target.value)}
                className="mt-1 block w-full px-1 py-2 text-base border border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
              >
                <option value="all">All Time</option>
                <option value="year">Year</option>
                <option value="month">Month</option>
                <option value="week">Week</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700">Sort By</label>
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value)}
                className="mt-1 block w-full px-1 py-2 text-base border border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
              >
                <option value="top">Top</option>
                <option value="controversial">Controversial</option>
              </select>
            </div>

            <button
            onClick={() => {
              setIsLoading(true);
              fetchSubredditAnalysis(name || "", timeFilter, sortBy)
                .then(data => {
                  setAnalysis(data);
                  setIsLoading(false);
                })
                .catch(err => {
                  setError(err instanceof Error ? err.message : "An unknown error occurred");
                  setIsLoading(false);
                });
            }}
            className="block self-end px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 hover:cursor-pointer focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
          >
            Analyze
          </button>
          </div>
        </div>

        {isLoading && (
          <div className="text-center py-8">
            <p>Loading analysis...</p>
          </div>
        )}

        {error && (
          <div className="bg-red-50 border-l-4 border-red-400 p-4 mt-4">
            <div className="flex">
              <div className="ml-3">
                <p className="text-sm text-red-700">
                  Error: {error}
                </p>
              </div>
            </div>
          </div>
        )}

        {!isLoading && !error && analysis && (
          <div className="mt-4">
            {renderNGrams()}
            {renderNamedEntities()}
          </div>
        )}
      </div>
    </Template>
  );
}
