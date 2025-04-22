import { SubredditAnalysis } from "../../types/redditAnalysis";
import ButtonPopover from "../ButtonPopover";

interface WordCloudProps {
  analysis: SubredditAnalysis;
}

// Component to display word cloud with description
export const WordCloud: React.FC<WordCloudProps> = ({ analysis }) => {
  const wordCloud = new Image();
  wordCloud.src = `data:image/png;base64, ${analysis.top_named_entities_wordcloud}`;

  return (
    <div className="flex flex-col items-center">
      <h1 className="font-bold text-xl text-center mt-4 mb-4 p-1">
        Word Cloud
      </h1>
      <ButtonPopover title="What's a Word Cloud?'">
        A word cloud is a visual representation of the most popular words in the
        subreddit. The size of the word corresponds to its frequency in the
        subreddit.
      </ButtonPopover>
      <div className="mt-4 p-2 pl-6 pr-6 pb-6 bg-white w-full mx-auto">
        {analysis?.top_named_entities_wordcloud ? (
          <img
            className="mx-auto w-[100%] md:w-[50%] h-auto"
            src={wordCloud.src}
            alt="Word Cloud"
          />
        ) : (
          <p className="text-gray-400 p-4">No Word Cloud</p>
        )}
      </div>
    </div>
  );
};
