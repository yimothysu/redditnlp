import React, { useState, useEffect } from "react";
import Template from "./Template.tsx";
import { SubredditCard } from "../src/components/SubredditCard.tsx";
import { useNavigate } from "react-router-dom";
import { fetchPopularSubreddits, PopularSubredditsResponse } from "../src/lib/api.ts";

function PopularCommunities() {
  const [popularCommunities, setPopularCommunities] = useState<PopularSubredditsResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const communities = await fetchPopularSubreddits();
        setPopularCommunities(communities);
        setLoading(false);
      } catch (err) {
        setError("Failed to fetch popular communities.");
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return <div>Loading...</div>;
  }

  if (popularCommunities){
    return (
      <div className="bg-gray-200 mt-6 p-4">
        <h1 className="font-bold text-xl mb-1">
          Explore NLP For Popular Communities
        </h1>
        <p>
          Below are some of the most popular subreddit communities right now.
          Explore NLP for them.
        </p>
        <div className="flex flex-wrap gap-4 mt-4">
          {popularCommunities.reddits.map((subreddit) => (
            <SubredditCard key={subreddit.name} subredditName={subreddit.name} />
          ))}
        </div>
      </div>
    );
  } else if (error){
      return <div>{error}</div>;
  }
}

export default function Home() {
  const [subreddit, setSubreddit] = useState("");

  const navigate = useNavigate();

  const handleSubredditInputBoxChange = (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    setSubreddit(e.target.value);
  };

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    navigate(`/subreddit/${subreddit}`);
  };

  return (
    <Template>
      <div className="flex flex-col items-center mt-10">
        <div className="flex rounded-full bg-[#a5bf5d] text-white font-bold text-3xl h-24 w-24 justify-center items-center">
          NLP
        </div>
        <h1 className="pt-2 pb-4 text-3xl">
          <b>Reddit</b>NLP
        </h1>
        <form onSubmit={(e) => handleSubmit(e)} className="text-center">
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
        </form>
      </div>
      <PopularCommunities />
    </Template>
  );
}
