import { ChangeEvent } from "react";
import { useState } from "react";
import { useParams } from "react-router-dom";
import RedditAnalysisDisplay from "../Components/RedditAnalysisDisplay.tsx";
import Template from "../pages/Template.tsx";
import { Helmet } from "react-helmet";

const subreddits = [
  "AskReddit",
  "politics",
  "AskOldPeople",
  "gaming",
  "science",
  "popculturechat",
  "worldnews",
  "technology",
  "100YearsAgo",
  "Feminism",
  "unpopularopinion",
  "philosophy",
  "mentalhealth",
  "teenagers",
  "AskMen",
  "AskWomen",
  "personalfinance",
  "changemyview",
  "LateStageCapitalism",
  "UpliftingNews",
];

export default function RedditPage() {
  const { name } = useParams();
  const [subreddit, setSubreddit] = useState("");

  const handleSubredditChange = (e: ChangeEvent<HTMLSelectElement>) => {
    setSubreddit(e.target.value);
  };

  const renderSelectReddit = () => {
    return (
      <div className="flex items-center mr-5">
        <p className="text-black font-medium mr-2">Compare r/{name} to: </p>
        <select
          className="ml-2 bg-white border-2 border-black rounded-lg p-2 shadow-sm"
          onChange={handleSubredditChange}
          value={subreddit}
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

  let displayFull = !subreddit ? "w-full" : "w-full md:w-1/2";

  const Analysis = ({inComparisonMode}) => {
    return (
      <div className="flex w-full">
        <div className={displayFull}>
          <RedditAnalysisDisplay name={name} inComparisonMode={inComparisonMode} />
        </div>
        {subreddit && (
          <div className="w-1/2 hidden md:block">
            {<RedditAnalysisDisplay name={subreddit} inComparisonMode={inComparisonMode} />}
          </div>
        )}
      </div>
    );
  };

  return (
    <>
      <Helmet>
        <title>{`r/${name} | Reddit NLP`}</title>
        <meta
          name="description"
          content={`Explore natural language trends in r/${name}${
            subreddit ? ` and compare with r/${subreddit}` : ""
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
            name: `Reddit NLP - r/${name}`,
            url: `https://redditnlp.com/subreddit/r/${name}`,
            description: `Explore NLP trends in r/${name}${
              subreddit ? ` and compare with r/${subreddit}` : ""
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
        {subreddit && (
          <div className="hidden md:block bg-white w-[80%] mx-auto p-3 mt-4 rounded-sm shadow-sm">
            <h1 className="text-center text-2xl">
              r/{name} vs. r/{subreddit}
            </h1>
            <p className="text-center text-gray-500">
              Below is an in-depth analysis comparing r/{name} and r/{subreddit}
            </p>
          </div>
        )}
        {/* <div className="flex w-full">
        <div className={displayFull}>
          <RedditAnalysisDisplay name={name}/>
        </div>
        {subreddit && <div className="w-1/2 hidden md:block">
          {<RedditAnalysisDisplay name={subreddit}/>}
        </div>
        }     
      </div> */}
        <Analysis inComparisonMode={subreddit}></Analysis>
      </Template>
    </>
  );
}
