import React from 'react';
import Template from './Template.tsx';


import { useState } from 'react';

export default function Home() {

    const [subreddit_input_box, set_subreddit_input_box] = useState("");

    const handleSubredditInputBoxChange = (
      e: React.ChangeEvent<HTMLInputElement>
    ) => {
      set_subreddit_input_box(e.target.value);
    };
  
    const checkIsValidReddit = () => {
      const subreddit_regex = /^r\/\w+/;
      if (subreddit_regex.test(subreddit_input_box)) {
        return true;
      } else {
        return false;
      }
    };
	
    return (
        <Template>
            <div className="header-container bg-gray-100">
            <div className="rounded-full bg-lime-400 text-white font-bold text-xl p-8 mt-8">
                NLP
            </div>
            <h1 className="pt-2 pb-4">
                <b>Reddit</b>NLP
            </h1>
            <input
                type="text"
                placeholder="enter a community (ex: r/cats)"
                value={subreddit_input_box}
                onChange={handleSubredditInputBoxChange}
                className={`subreddit-input-box bg-white rounded-full p-2 w-96 ${checkIsValidReddit() ? "border-red" : "border-green"}`}
            />
            <button
                className="rounded-full bg-emerald-400 py-3 px-8 mt-4 text-white font-bold"
                onClick={checkIsValidReddit}
            >
                Explore NLP
            </button>
            </div>
        </Template>
      );
}