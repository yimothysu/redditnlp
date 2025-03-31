import {ChangeEvent} from "react";
import { useState } from "react";
import { useParams } from "react-router-dom";
import RedditAnalysisDisplay from "../Components/RedditAnalysisDisplay.tsx";
import Template from "../pages/Template.tsx";

const subreddits = [
  'AskReddit', 
  'politics', 
  'AskOldPeople',
  'gaming', 
  'science', 
  'popculturechat',
  'worldnews',
  'technology', 
  '100YearsAgo',
  'Feminism',
  'unpopularopinion', 
  'philosophy',
  'mentalhealth', 
  'teenagers',
  'AskMen',
  'AskWomen', 
  'personalfinance', 
  'changemyview',
  'LateStageCapitalism', 
  'UpliftingNews'];


export default function RedditPage() {
  const { name } = useParams();
  const [subreddit, setSubreddit] = useState("");

  const handleSubredditChange = (e: ChangeEvent<HTMLSelectElement>) => {
    setSubreddit(e.target.value);
  }

  const renderSelectReddit = () => {
    return (
      <div className="flex items-center mr-5">
        <p className="text-black font-medium mr-2">Compare r/{name} to: </p>
        <select 
          className="ml-2 bg-white border-2 border-black rounded-lg p-2 shadow-sm"
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
  
  const Analysis = () => {
    if(subreddit != "") {
      return (
        <div className="flex w-full">
        <div className={displayFull}>
          <RedditAnalysisDisplay name={name} inComparisonMode="true"/>
        </div>
        {subreddit && <div className="w-1/2">
          {<RedditAnalysisDisplay name={subreddit} inComparisonMode="true"/>}
        </div>
        }     
      </div>
      );
    }
    
    return (
      <div className="flex w-full">
      <div className={displayFull}>
        <RedditAnalysisDisplay name={name}  inComparisonMode="false"/>
      </div>
      {subreddit && <div className="w-1/2">
        {<RedditAnalysisDisplay name={subreddit}  inComparisonMode="false"/>}
      </div>
      }     
    </div>
    );
  }

  return (
    <Template>
      <div className="flex">
        <div className="ml-auto">
          {renderSelectReddit()}
        </div>
      </div>
      {subreddit && 
        <div className="bg-gray-200 w-[50%] mx-auto p-3 mt-4 rounded-sm ">
          <h1 className="text-center text-black font-bold mb-2 text-2xl">r/{name} vs. r/{subreddit}</h1>
          <p className="text-center text-black">Below is an in-depth analysis comparing r/{name} and r/{subreddit}</p>
        </div>}
      {/* <div className="flex w-full">
        <div className={displayFull}>
          <RedditAnalysisDisplay name={name}/>
        </div>
        {subreddit && <div className="w-1/2">
          {<RedditAnalysisDisplay name={subreddit}/>}
        </div>
        }     
      </div> */}
      <Analysis></Analysis>
    </Template>
  );
}
