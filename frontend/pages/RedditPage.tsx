import { ChangeEvent } from "react";
import { useState } from "react";
import { useParams } from "react-router-dom";
import RedditAnalysisDisplay from "../Components/RedditAnalysisDisplay.tsx";
import Template from "../pages/Template.tsx";
import { Helmet } from "react-helmet";
import {subreddits} from '../subreddits.tsx';

// RedditPage is the page that displays when /subreddit/{name} is visited
export default function RedditPage() {

  const { subredditName } = useParams();
  const [compareSubredditName, setCompareSubredditName] = useState("");
  const [inComparisonMode, setInComparisonMode] = useState("false")

  // When the user selects a subreddit to compare to, update state to reflect that
  const handleSubredditChange = (e: ChangeEvent<HTMLSelectElement>) => {
    setCompareSubredditName(e.target.value);
  };

  // Select subreddit dropdown menu
  const renderSelectReddit = () => {
    console.log('subreddit: ', compareSubredditName)
    return (
      <div className="flex items-center mr-5">
        <p className="text-black font-medium mr-2">Compare r/{subredditName} to: </p>
        <select
          className="ml-2 bg-white border-2 border-black rounded-lg p-2 shadow-sm"
          onChange={handleSubredditChange}
          value={compareSubredditName}
        >
          <option key="" value="">
            Select Subreddit
          </option>
          {subreddits.map((r) => (
            <option key={r} value={r}>
              {r}
            </option>
          ))}
        </select>
      </div>
    );
  };

  // Depending on whether a compare subreddit is selected, adjust size of reddit analyses component
  let displayFull = !compareSubredditName ? "w-full" : "w-full md:w-1/2";

  // Displays side by side analysis of subreddits
  const Analysis = () => {
    if (compareSubredditName != "" && compareSubredditName != "Select Subreddit") {
      setInComparisonMode("true")
    }
    else {
      setInComparisonMode("false")
    }

    return (
      <div className="flex w-full">
        <div className={displayFull}>
          <RedditAnalysisDisplay name={subredditName} inComparisonMode={inComparisonMode} />
        </div>
        {compareSubredditName && (
          <div className="w-1/2 hidden md:block">
            {<RedditAnalysisDisplay name={compareSubredditName} inComparisonMode={inComparisonMode} />}
          </div>
        )}
      </div>
    );
  };

  return (
    <>
      <Helmet>
        <title>{`r/${subredditName} | Reddit NLP`}</title>
        <meta
          name="description"
          content={`Explore natural language trends in r/${subredditName}${
            compareSubredditName ? ` and compare with r/${compareSubredditName}` : ""
          }.`}
        />
        <meta
          name="keywords"
          content="Reddit, NLP, Natural Language Processing"
        />
        <link
          rel="canonical"
          href={`https://redditnlp.com/subreddit/r/${encodeURIComponent(
            name || ""
          )}`}
        />
        <script type="application/ld+json">
          {JSON.stringify({
            "@context": "https://schema.org",
            "@type": "WebPage",
            name: `Reddit NLP - r/${subredditName}`,
            url: `https://redditnlp.com/subreddit/r/${subredditName}`,
            description: `Explore NLP trends in r/${subredditName}${
              compareSubredditName ? ` and compare with r/${compareSubredditName}` : ""
            }.`,
          })}
        </script>
      </Helmet>
      <Template>
        <div className="hidden md:block">
          <div className="flex">
            <div className="ml-auto">{renderSelectReddit()}</div>
          </div>
        </div>
        {compareSubredditName && (
          <div className="hidden md:block bg-white w-[80%] mx-auto p-3 mt-4 rounded-sm shadow-sm">
            <h1 className="text-center text-2xl">
              r/{subredditName} vs. r/{compareSubredditName}
            </h1>
            <p className="text-center text-gray-500">
              Below is an in-depth analysis comparing r/{subredditName} and r/{compareSubredditName}
            </p>
          </div>
        )}
        <Analysis></Analysis>
      </Template>
    </>
  );
}
