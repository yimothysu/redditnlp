import {ChangeEvent} from "react";
import { useState } from "react";
import { useParams } from "react-router-dom";
import RedditAnalysisDisplay from "../Components/RedditAnalysisDisplay.tsx";
import Template from "../pages/Template.tsx";

// TODO: Replace list with list of cached subreddits
const subreddits = ["AskReddit", "ufl", "teenagers"];


export default function RedditPage() {
  const { name } = useParams();
  const [subreddit, setSubreddit] = useState("");

  const handleSubredditChange = (e: ChangeEvent<HTMLSelectElement>) => {
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
