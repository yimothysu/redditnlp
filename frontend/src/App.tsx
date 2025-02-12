import { useState } from "react";
import "./App.css";
//import axios from 'axios';

import Home from '../pages/Home.tsx'
import RedditPage from '../pages/RedditPage.tsx'
import CompareRedditPage from '../pages/CompareRedditPage.tsx'


import { BrowserRouter, Routes, Route } from "react-router-dom";


function App() {
  const [subreddit_input_box, set_subreddit_input_box] = useState("");

  const handleSubredditInputBoxChange = (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    set_subreddit_input_box(e.target.value);
  };

  const generateSubredditAnalysis = () => {
    const subreddit_regex = /^r\/\w+/;
    if (subreddit_regex.test(subreddit_input_box)) {
      console.log("valid subreddit format");
    } else {
      console.log("invalid subreddit format");
    }

    // const api_endpoint = 'http://localhost:8000/sample/' + subreddit_input_box
    // axios.get(api_endpoint)
  };

  // <input type='text' value={subreddit_input_box} onChange={handleSubredditInputBoxChange} className="subreddit-input-box"/>
  return (
  <BrowserRouter>
    <Routes>
      <Route path="/" element={<Home/>}></Route>
      <Route path="/reddit-page" element={<RedditPage/>}></Route>
      <Route path="/compare-reddit-page" element={<CompareRedditPage/>}></Route>
    </Routes>
  </BrowserRouter>
  )
  /*return (
    <>
      <div>
        <a href="https://vite.dev" target="_blank">
          <img src={viteLogo} className="logo" alt="Vite logo" />
        </a>
        <a href="https://react.dev" target="_blank">
          <img src={reactLogo} className="logo react" alt="React logo" />
        </a>
      </div>
      <h1>Vite + React</h1>
      <div className="card">
        <button onClick={() => setCount((count) => count + 1)}>
          count is {count}
        </button>
        <p>
          Edit <code>src/App.tsx</code> and save to test HMR
        </p>
      </div>
      <p className="read-the-docs">
        Click on the Vite and React logos to learn more
      </p>
    </>
  )
}

export default App;
