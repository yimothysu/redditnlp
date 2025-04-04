import "./App.css";
//import axios from 'axios';

import Home from "../pages/Home.tsx";
import RedditPage from "../pages/RedditPage.tsx";

import { Routes, Route } from "react-router-dom";

function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />}></Route>
      <Route path="/subreddit/:name" element={<RedditPage />}></Route>
    </Routes>
  );
}

export default App;
