import { SubredditAnalysis } from "../../types/redditAnalysis";
import ButtonPopover from "../ButtonPopover";
import WordEmbeddingsGraph from "../WordEmbeddingsGraph";

interface WordEmbeddingsProps {
  analysis: SubredditAnalysis;
  inComparisonMode: string;
}

export const WordEmbeddings: React.FC<WordEmbeddingsProps> = ({
  analysis,
  inComparisonMode,
}) => {
  let embeddings = analysis.top_named_entities_embeddings;
  let embeddingsDisplay =
    embeddings &&
    Object.entries(embeddings).map(([word, [x, y]]) => ({
      word,
      x: x * 5,
      y: y * 5,
    }));

  return (
    <div
      className="flex flex-col items-center"
      style={{ width: "100%", height: "auto" }}
    >
      <h1 className="font-bold text-xl mt-4 text-center p-1 mb-4">
        Word Embeddings
      </h1>
      <ButtonPopover title="What are word embeddings?">
        Word embeddings are vector representations of words in high-dimensional
        space, offering insights into meaning and context. Below is a 2D
        projection of the most popular words in the subreddit (reduced via PCA).
      </ButtonPopover>
      <div className="mt-4 p-2 pl-6 pr-6 pb-6 bg-white">
        <div className="bg-gray-100 rounded">
          {embeddingsDisplay && embeddingsDisplay.length > 0 ? (
            <WordEmbeddingsGraph
              embeddings={embeddingsDisplay}
              isComparisonMode={inComparisonMode}
            />
          ) : (
            <p className="text-gray-400 p-4">No Embeddings</p>
          )}
        </div>
      </div>
    </div>
  );
};
