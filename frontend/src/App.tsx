import { useState } from "react";
import "./App.css";
import axios from 'axios';

function App() {
  const [subreddit_input_box, set_subreddit_input_box] = useState('what subreddit do you want to analyze? (Ex: r/cats)')

  const handleSubredditInputBoxChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      set_subreddit_input_box(e.target.value);
  };

  const generateSubredditAnalysis = () => {
    const subreddit_regex = /^r\/\w+/;
    if(subreddit_regex.test(subreddit_input_box)) {
      console.log('valid subreddit format')
    }
    else {
      console.log('invalid subreddit format')
    }

    const api_endpoint = 'http://localhost:8000/sample/' + subreddit_input_box
    axios.get(api_endpoint)
  };

  // <input type='text' value={subreddit_input_box} onChange={handleSubredditInputBoxChange} className="subreddit-input-box"/>
  return (
    <div className='header-container'>
      <h1>RedditNLP</h1>
      <div className='input_and_button'>
        <input type='text' value={subreddit_input_box} onChange={handleSubredditInputBoxChange} className="subreddit-input-box"/>
        <button className='analyze-button' onClick={generateSubredditAnalysis}>analyze</button>
      </div>
    </div>
  );
}

export default App;
