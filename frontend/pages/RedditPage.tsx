import React from "react";
import { useState, useEffect, useRef } from "react";
import { useParams } from "react-router-dom";
import RedditAnalysisDisplay from "../Components/RedditAnalysisDisplay.tsx";
import Template from "../pages/Template.tsx";

const renderGenerationTimeEstimatesTable = () => {
  return (
    <div className="overflow-x-auto flex items-center justify-center">
      <table className="table-auto border-collapse text-sm m-2.5">
        <thead>
          <tr>
            <th className="px-6 py-2 text-center text-left">Time Filter</th>
            <th className="px-4 py-2 text-center text-left">
              # of Posts Sampled
            </th>
            <th className="px-6 py-2 text-center text-left">
              Generation Time
            </th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="px-4 py-1.5 text-center bg-gray-200">Week</td>
            <td className="px-4 py-1.5 text-center bg-gray-200">50</td>
            <td className="px-4 py-1.5 text-center bg-gray-200">~3-5 min</td>
          </tr>
          <tr>
            <td className="px-4 py-1.5 text-center">Month</td>
            <td className="px-4 py-1.5 text-center">100</td>
            <td className="px-4 py-1.5 text-center">~5-7 min</td>
          </tr>
          <tr>
            <td className="px-4 py-1.5 text-center bg-gray-200">Year</td>
            <td className="px-4 py-1.5 text-center bg-gray-200">200</td>
            <td className="px-4 py-1.5 text-center bg-gray-200">~10 min</td>
          </tr>
          <tr>
            <td className="px-4 py-1.5 text-center">All Time</td>
            <td className="px-4 py-1.5 text-center">350</td>
            <td className="px-4 py-1.5 text-center">~15 min</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
};

// TODO: Replace list with list of cached subreddits
const subreddits = ["AskReddit", "ufl", "teenagers"];


export default function RedditPage() {
  const { name } = useParams();
  const [subreddit, setSubreddit] = useState("");

  const handleSubredditChange = (e) => {
    setSubreddit(e.target.value);
  }

  const renderSelectReddit = () => {
    return (
      <div className="flex items-center mr-5">
        <p className="text-gray-800 font-medium mr-2">Compare r/{name} to Another Subreddit: </p>
        <select 
          className="ml-2 bg-white border-gray-500 rounded-lg p-2 shadow-sm"
          onChange={handleSubredditChange}
          value={subreddit}>
            <option key="" value="">Select Subreddit</option>
            {subreddits.map(r => (
              <option key={r} value={r}>{r}</option>
            ))}
        </select>
      </div>
    )
    }

  let displayFull = !subreddit ? "w-full" : "w-1/2";

  return (
    <Template>
      <div className="flex">
        <div className="ml-auto">
          {renderSelectReddit()}
        </div>
      </div>
      {subreddit && 
        <div className="bg-white w-[80%] mx-auto p-3 mt-4 rounded-sm shadow-sm">
          <h1 className="text-center text-2xl">r/{name} vs. r/{subreddit}</h1>
          <p className="text-center text-gray-500">Below is an in-depth analysis comparing r/{name} and r/{subreddit}</p>
        </div>}
      <div className="flex w-full">
        <div className={displayFull}>
          <RedditAnalysisDisplay name={name}/>
        </div>
        {subreddit && <div className="w-1/2">
          {<RedditAnalysisDisplay name={subreddit}/>}
        </div>
        }     
      </div>
    </Template>
  );
}
