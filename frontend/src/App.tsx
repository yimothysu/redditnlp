import { useState } from "react";
import "./App.css";
//import axios from 'axios';

import Home from "../pages/Home.tsx";
import RedditPage from "../pages/RedditPage.tsx";
import CompareRedditPage from "../pages/CompareRedditPage.tsx";

import { Routes, Route } from "react-router-dom";

function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />}></Route>
      <Route path="/subreddit/:name" element={<RedditPage />}></Route>
      <Route
        path="/compare-reddit-page"
        element={<CompareRedditPage />}
      ></Route>
    </Routes>
  );
}

export default App;
