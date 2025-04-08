//import React, { useState, useEffect } from "react";
import Template from "./Template.tsx";
import { SubredditCard } from "../src/components/SubredditCard.tsx";
// import { useNavigate } from "react-router-dom";
// import { fetchPopularSubreddits, PopularSubredditsResponse } from "../src/lib/api.ts";
import { Helmet } from "react-helmet";
import { useState } from "react";
import { requestSubreddit } from "../src/lib/api.ts";
import {subreddits} from '../subreddits.tsx';

function SubredditsCached() {

    return (
        <>
            <Helmet>
                <title>Reddit NLP</title>
                <meta
                    name="description"
                    content="Explore NLP for Reddit communities"
                />
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
            <div className="bg-gray-100 p-7">
                <h1 className="font-bold text-xl mb-1">
                    Explore NLP For Popular Communities
                </h1>
                <p>
                    Below are some of the most popular subreddit communities
                    right now. Explore NLP for them.
                </p>
                <div className="flex flex-wrap gap-5 mt-4">
                    {subreddits.map((subreddit) => (
                        <SubredditCard
                            key={subreddit}
                            subredditName={subreddit}
                        />
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
            const response = await requestSubreddit(
                subreddit.trim(),
                email.trim()
            );

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
            <h1 className="font-bold text-xl mb-1">Submit a Request</h1>
            <h1 className="mb-4">
                {" "}
                Have a subreddit you want to analyze that's not on our website?
                Submit up to one request/day.
            </h1>

            {error && <div className="text-red-600 mb-2">{error}</div>}
            {submitted && (
                <div className="text-green-600 mb-2">
                    Request submitted successfully!
                </div>
            )}

            <div className="relative mb-4">
                <div className="absolute pl-4 flex items-center inset-y-0 text-xl">
                    r/
                </div>
                <input
                    type="text"
                    value={subreddit}
                    onChange={(e) => setSubreddit(e.target.value)}
                    placeholder="enter a community"
                    className={`bg-white rounded-lg p-2 pl-8 w-70 shadow outline outline-2 outline-black`}
                />
            </div>
            <div className="relative">
                <input
                    type="text"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="enter your email"
                    className={`bg-white rounded-lg p-2 pl-8 w-70 shadow outline outline-2 outline-black`}
                />
            </div>
            <input
                type="submit"
                value="Submit"
                className="rounded-xl bg-[#57bcb3] py-2 px-7 mt-7 text-white font-bold shadow hover:cursor-pointer hover:bg-[#58ada6]"
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
                <div className="flex rounded-full bg-[#a5bf5d] text-white font-bold text-3xl h-24 w-24 justify-center items-center">
                    NLP
                </div>
                <h1 className="pt-2 pb-4 text-3xl">
                    <b>Reddit</b>NLP
                </h1>
            </div>
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
