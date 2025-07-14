import { SubredditAnalysis, NGram } from "../../types/redditAnalysis";
import ButtonPopover from "../ButtonPopover";

interface NGramsProps {
  analysis: SubredditAnalysis;
  timeFilter: string;
}

const cuteColors = [
  'text-pink-400', 'text-sky-400', 'text-purple-400', 'text-green-400',
  'text-blue-400', 'text-fuchsia-400', 'text-indigo-400',
  'text-amber-400',
];

export const NGrams: React.FC<NGramsProps> = ({ analysis }) => {
  return (
    <div className="mt-7">
      <div className="flex flex-col items-center">
        <h2 className="text-xl text-center font-bold">
          Most Mentioned Bi-Grams
        </h2>
        <br />
        <ButtonPopover title="What's a Bi-Gram?">
          A Bi-Gram is 2 consecutive words in a text. For example, the sentence
          "I love pizza" has 2 bigrams: "I love" and "love pizza".
        </ButtonPopover>
      </div>
      <div
        style={{
          marginTop: "0px",
          marginBottom: "40px",
          marginLeft: "25px",
          marginRight: "25px",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          position: "relative",
        }}
      ></div>
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5">
        {analysis.top_n_grams.slice().reverse().map(
          (ngram: NGram, idx: number) =>
            ngram.name.length > 0 && (
              <div className="mb-4 text-center">
                <div className={"text-md font-normal " + cuteColors[idx % cuteColors.length]} key={idx}>
                    {ngram.name}
                </div>
                <div className="text-xs text-gray-500">
                  count: {ngram.count}
                </div>
                <hr className="border border-gray-200 mb-2 mt-2"></hr>
              </div>
            )
        )}
      </div>
    </div>
  );
};
