import { SubredditAnalysis, NGram } from "../../types/redditAnalysis";
import ButtonPopover from "../ButtonPopover";
import { formatDate } from "../../utils/dateUtils";

interface NGramsProps {
  analysis: SubredditAnalysis;
  timeFilter: string;
}

interface NGramsForDateProps {
  date: string;
  ngrams: NGram[];
  timeFilter: string;
}

const NGramsForDate: React.FC<NGramsForDateProps> = ({
  date,
  ngrams,
  timeFilter,
}) => {
  return (
    <div
      key={date}
      className="flex flex-col h-full mb-3 mr-3 ml-3 border bg-white border-gray-200 rounded-l shadow-md"
    >
      <h3 className="text-lg text-center p-1 font-semibold bg-gray-100">
        {formatDate(date, timeFilter)}
      </h3>
      <div className="flex font-semibold pt-1 pb-1 pr-5 pl-5 bg-indigo-200 justify-between">
        <h2>Word</h2>
        <h2>Count</h2>
      </div>
      <ul className="list-none pl-5 pr-5 pt-2">
        {ngrams.map((ngram: NGram, index: number) => (
          <li key={index} className="pb-1 pt-1">
            <div className="text-[14px] flex justify-between">
              <div>{ngram[0]}</div>
              <div>{ngram[1]}</div>
            </div>
            <hr className="border-t border-gray-300 my-1" />
          </li>
        ))}
      </ul>
    </div>
  );
};

export const NGrams: React.FC<NGramsProps> = ({ analysis, timeFilter }) => {
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
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4">
        {Object.entries(analysis.top_n_grams).map(
          ([date, ngrams]) =>
            ngrams.length > 0 && (
              <div className="mb-4" key={date}>
                <NGramsForDate
                  date={date}
                  ngrams={ngrams}
                  timeFilter={timeFilter}
                />
              </div>
            )
        )}
      </div>
    </div>
  );
};
