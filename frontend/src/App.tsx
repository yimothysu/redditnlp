import Home from "../pages/Home.tsx";
import RedditPage from "../pages/RedditPage.tsx";

import { Routes, Route } from "react-router-dom";

// Based on url path, display components for Home and analysis display
function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />}></Route>
      <Route path="/subreddit/:subredditName" element={<RedditPage />}></Route>
    </Routes>
  );
}

export default App;
