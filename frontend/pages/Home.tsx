//import React, { useState, useEffect } from "react";
import Template from "./Template.tsx";
import { SubredditCard } from "../src/components/SubredditCard.tsx";
// import { useNavigate } from "react-router-dom";
// import { fetchPopularSubreddits, PopularSubredditsResponse } from "../src/lib/api.ts";
import { Helmet } from "react-helmet";
import { useState } from "react";
import { requestSubreddit } from "../src/lib/api.ts";
import { category_to_subreddits } from "../src/constants/subreddits.ts";

function SubredditsCached() {
  return (
    <>
      <Helmet>
        <title>Reddit NLP</title>
        <meta name="description" content="Explore NLP for Reddit communities" />
        <meta
          name="keywords"
          content="Reddit, NLP, Natural Language Processing"
        />
        <link rel="canonical" href="https://redditnlp.com" />
        <script type="application/ld+json">
          {JSON.stringify({
            "@context": "https://schema.org",
            "@type": "WebSite",
            name: "Reddit NLP",
            url: "https://redditnlp.com",
            description:
              "Reddit NLP helps you explore natural language trends in Reddit communities.",
          })}
        </script>
      </Helmet>
      <div className="bg-gray-100 p-4 md:p-7">
        <div className="flex gap-4 items-center justify-center">
          <h1 className="font-bold text-xl mb-1 mt-4 text-center">
            The most comprehensive subreddit NLP analysis on the internet 
          </h1>
        </div>
        <p className="pt-2 text-center">
          Below are some of the most popular subreddit communities right now.
          Explore NLP for them.
        </p>
        <div className="">
          {Object.entries(category_to_subreddits).map(([category, subreddits_in_category]) => (
            <div className="pt-8">
              <hr className="border-1 border-white"></hr>
              <div className="gap-4 items-center justify-center text-center flex flex-row bg-gray-200 rounded-sm shadow-sm mx-auto pb-2 pt-2">
                <h1 className="mt-2 tracking-wide font-semibold text-md">{category}</h1>
                <img
                  src={"category_pics/" + category.toLowerCase() + ".png"}
                  className="rounded-sm w-8 bg-gray-200"
                />
              </div>
              <hr className="border-1 border-white"></hr>
              <div className="mt-3 flex flex-wrap gap-3 md:gap-5">
                {subreddits_in_category.map((subreddit) => (
                <SubredditCard key={subreddit} subredditName={subreddit} />
                ))}
              </div>
            </div>
          ))}
        </div>
        </div>
    </>
  );
}

function RequestAnalysisForSpecificSubreddit() {
  const [subreddit, setSubreddit] = useState("");
  const [email, setEmail] = useState("");
  const [submitted, setSubmitted] = useState(false);
  const [error, setError] = useState("");

  const submitRequest = async () => {
    if (submitted) {
      return;
    }

    setError("");
    setSubmitted(false);

    if (!subreddit.trim()) {
      setError("Please enter a subreddit name.");
      return;
    }
    if (!email.trim() || !/^[\w-.]+@([\w-]+\.)+[\w-]{2,4}$/.test(email)) {
      setError("Please enter a valid email address.");
      return;
    }

    try {
      const response = await requestSubreddit(subreddit.trim(), email.trim());

      if (response.ok) {
        setSubmitted(true);
        setSubreddit("");
        setEmail("");
      } else {
        setError("Something went wrong. Please try again later.");
      }
    } catch (err) {
      setError("Network error. Please try again.");
    }
  };

  return (
    <div className="p-7 flex flex-col justify-center items-center">
      <h1 className="font-bold text-xl mt-5 mb-4">Submit a Request</h1>
      <img src={"email.png"} className="w-13 mb-3 mt-1"/>
      <h1 className="mb-4">
        {" "}
        Have a subreddit you want to analyze that's not on our website? Submit
        up to one request/day.
      </h1>
      {error && <div className="text-red-600 mb-2">{error}</div>}
      {submitted && (
        <div className="text-green-600 mb-2">
          Request submitted successfully!
        </div>
      )}

      <div className="relative mb-1 mt-2">
        <div className="absolute pl-4 flex items-center inset-y-0 text-xl">
          r/
        </div>
        <input
          type="text"
          value={subreddit}
          onChange={(e) => setSubreddit(e.target.value)}
          placeholder="enter a community"
          className={`bg-white rounded-lg p-2 pl-8 w-70`}
        />
      </div>
      <input
        type="text"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        placeholder="enter your email"
        className="bg-white rounded-lg p-2 pl-8 w-70 mt-2"
      />
      <input
        type="submit"
        value="Submit"
        className="rounded-lg bg-[#fa6f4d] py-2 px-7 mt-4 text-white font-bold shadow hover:cursor-pointer hover:bg-[#e36748]"
        onClick={submitRequest}
      />
    </div>
  );
}

export default function Home() {
  // };

  return (
    <Template>
      <div className="flex flex-col items-center pb-3">
        <div className="flex rounded-full bg-[#fa6f4d] text-white font-bold text-3xl h-24 w-24 justify-center items-center">
          NLP
        </div>
        <h1 className="pt-2 pb-8 text-3xl">
          <b>Reddit</b>NLP
        </h1>
      </div>
      <SubredditsCached></SubredditsCached>
      <RequestAnalysisForSpecificSubreddit></RequestAnalysisForSpecificSubreddit>
      <div className="flex gap-5 justify-end mr-5">
        <img
          src={"github_logo.png"}
          className="rounded-full w-9 h-9 object-cover cursor-pointer transition active:scale-95 active:brightness-90"
          onClick={() =>
            window.open("https://github.com/yimothysu/redditnlp", "_blank")
          }
        />
      </div>
      <div className="w-full h-4"></div>
    </Template>
  );
}
