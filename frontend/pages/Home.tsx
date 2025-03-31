//import React, { useState, useEffect } from "react";
import Template from "./Template.tsx";
import { SubredditCard } from "../src/components/SubredditCard.tsx";
// import { useNavigate } from "react-router-dom";
// import { fetchPopularSubreddits, PopularSubredditsResponse } from "../src/lib/api.ts";

function SubredditsCached() {
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

    return (
      <div className="bg-gray-100 p-7">
        <h1 className="font-bold text-xl mb-1">
          Explore NLP For Popular Communities
        </h1>
        <p>
          Below are some of the most popular subreddit communities right now.
          Explore NLP for them.
        </p>
        <div className="flex flex-wrap gap-5 mt-4">
          {subreddits.map((subreddit) => (
            <SubredditCard key={subreddit} subredditName={subreddit} />
          ))}
        </div>
      </div>
    );

}

function RequestAnalysisForSpecificSubreddit() {
  return (
    <div className="p-7 flex flex-col justify-center items-center">
      <h1 className="font-bold text-xl mb-1">Submit a Request</h1>
      <h1 className="mb-4"> Have a subreddit you want to analyze that's not on our website? Submit up to one request/day.</h1>
      <div className="relative mb-4">
            <div className="absolute pl-4 flex items-center inset-y-0 text-xl">
              r/
            </div>
            <input
              type="text"
              placeholder="enter a community"
              className={`bg-white rounded-lg p-2 pl-8 w-70 shadow outline outline-2 outline-black`}
            />
      </div>
      <div className="relative">
            <input
              type="text"
              placeholder="enter your email"
              className={`bg-white rounded-lg p-2 pl-8 w-70 shadow outline outline-2 outline-black`}
            />
      </div>
      <input
            type="submit"
            value="Submit"
            className="rounded-xl bg-[#57bcb3] py-2 px-7 mt-7 text-white font-bold shadow hover:cursor-pointer hover:bg-[#58ada6]"
          />
      </div>
  );
}

// function PopularCommunities() {
//   const [popularCommunities, setPopularCommunities] = useState<PopularSubredditsResponse | null>(null);
//   const [loading, setLoading] = useState<boolean>(true);
//   const [error, setError] = useState<string | null>(null);

//   useEffect(() => {
//     const fetchData = async () => {
//       try {
//         const communities = await fetchPopularSubreddits();
//         setPopularCommunities(communities);
//         setLoading(false);
//       } catch (err) {
//         setError("Failed to fetch popular communities.");
//         setLoading(false);
//       }
//     };

//     fetchData();
//   }, []);

//   if (loading) {
//     return <div>Loading...</div>;
//   }

//   if (popularCommunities){
//     return (
//       <div className="bg-gray-200 mt-6 p-4">
//         <h1 className="font-bold text-xl mb-1">
//           Explore NLP For Popular Communities
//         </h1>
//         <p>
//           Below are some of the most popular subreddit communities right now.
//           Explore NLP for them.
//         </p>
//         <div className="flex flex-wrap gap-4 mt-4">
//           {popularCommunities.reddits.map((subreddit) => (
//             <SubredditCard key={subreddit.name} subredditName={subreddit.name} />
//           ))}
//         </div>
//       </div>
//     );
//   } else if (error){
//       return <div>{error}</div>;
//   }
// }

export default function Home() {
  // const [subreddit, setSubreddit] = useState("");

  // const navigate = useNavigate();

  // const handleSubredditInputBoxChange = (
  //   e: React.ChangeEvent<HTMLInputElement>
  // ) => {
  //   setSubreddit(e.target.value);
  // };

  // const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
  //   e.preventDefault();
  //   navigate(`/subreddit/${subreddit}`);
  // };

  return (
    <Template>
      <div className="flex flex-col items-center pb-3">
        <div className="flex rounded-full bg-[#a5bf5d] text-white font-bold text-3xl h-24 w-24 justify-center items-center">
          NLP
        </div>
        <h1 className="pt-2 pb-4 text-3xl">
          <b>Reddit</b>NLP
        </h1>
        {/* <form onSubmit={(e) => handleSubmit(e)} className="text-center">
          <div className="relative">
            <div className="absolute pl-4 flex items-center inset-y-0 text-xl">
              r/
            </div>
            <input
              type="text"
              placeholder="enter a community"
              value={subreddit}
              onChange={handleSubredditInputBoxChange}
              className={`bg-white rounded-full p-2 pl-8 w-96 shadow`}
            />
          </div>
          <input
            type="submit"
            value="Explore NLP"
            className="rounded-full bg-[#57bcb3] py-3 px-8 mt-4 text-white font-bold shadow hover:cursor-pointer hover:bg-[#58ada6]"
          />
        </form> */}
      </div>
      {/* <PopularCommunities /> */}
      {/* <hr className="border-gray-200"></hr> */}
      <SubredditsCached></SubredditsCached>
      <RequestAnalysisForSpecificSubreddit></RequestAnalysisForSpecificSubreddit>
      <div className="flex gap-5 justify-end">
        <img 
          src={"github_logo.png"}
          className="rounded-full w-9 h-9 object-cover"
        />
        <img 
          src={"discord_logo.png"}
          className="rounded-full w-9 h-9 mr-4 object-cover"
        />
      </div>
      <div className="w-full h-4"></div>
      {/* <hr className="border-gray-200"></hr> */}
    </Template>
  );
}
