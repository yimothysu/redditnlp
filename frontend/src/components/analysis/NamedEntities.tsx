import { useState } from "react";
import { SubredditAnalysis, NamedEntity } from "../../types/redditAnalysis";
import ButtonPopover from "../ButtonPopover";
import { ColorKey } from "./ColorKey";
import { formatDate } from "../../utils/dateUtils";
import { ChevronRight } from "lucide-react";

interface NamedEntitiesProps {
  analysis: SubredditAnalysis;
  timeFilter: string;
}

interface TopNamedEntitiesForDateProps {
  date: string;
  entities: NamedEntity[];
  timeFilter: string;
}

const SummarizedSentimentBulletPoints: React.FC<{
  summary: string;
}> = ({ summary }) => {
  if (!summary) return null;
  const sentences = summary
    .split("---")
    .filter((sentence: string) => sentence.trim() !== "");
  return (
    <ul className="list-disc pl-4">
      {sentences.map((sentence: string, index: number) => (
        <li key={index}>{sentence.trim()}</li>
      ))}
    </ul>
  );
};

const LinksForEntity = ({ entity }: { entity: NamedEntity }) => {
  if (Object.keys(entity[4]).length === 0) {
    return null;
  }
  return (
    <div className="mt-4">
      {Object.entries(entity[4]).map(([_, link]) => (
        <div
          onClick={() => window.open(link, "_blank")}
          className="bg-gray-50 hover:bg-gray-100 transition-colors duration-200 p-3 text-sm font-medium rounded-lg text-gray-700 text-center flex items-center justify-center cursor-pointer border border-gray-100"
        >
          Top Post discussing "{entity[0]}"
          <ChevronRight className="w-5 h-5 ml-2 text-gray-500" />
        </div>
      ))}
    </div>
  );
};

const TopNamedEntitiesForDate: React.FC<TopNamedEntitiesForDateProps> = ({
  date,
  entities,
  timeFilter,
}) => {
  return (
    <div
      key={date}
      className="flex-grow flex-1 mb-6 mr-1 ml-1 bg-white rounded-xl overflow-hidden border border-gray-200"
    >
      <h3
        className="text-xl font-semibold bg-gradient-to-r from-gray-50 to-gray-100 border-b border-gray-200"
        style={{
          padding: "12px",
          textAlign: "center",
        }}
      >
        {formatDate(date, timeFilter)}
      </h3>
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-3 gap-0.5 bg-gray-100">
        {entities.map((entity: NamedEntity, index: number) => {
          const backgroundColor =
            entity[2] >= 0.5 && entity[2] <= 1
              ? "#AFE1AF"
              : entity[2] >= 0.1 && entity[2] < 0.5
              ? "#e2f4a5"
              : entity[2] >= -0.1 && entity[2] < 0.1
              ? "#FFFFC5"
              : entity[2] >= -0.5 && entity[2] < -0.1
              ? "#FFD580"
              : entity[2] >= -1 && entity[2] < -0.5
              ? "#ffb9b9"
              : "bg-gray-100";
          return (
            <div
              key={index}
              className="bg-white p-5 transition-all duration-200 border border-gray-100 hover:border-gray-200"
            >
              <div className="flex justify-center mb-4">
                <h4 className="text-lg font-bold text-gray-800">
                  {entity[0]}
                </h4>
              </div>
              <div className="space-y-4">
                <div className="flex flex-col items-center">
                  <span
                    className="px-4 py-2 rounded-lg text-sm font-medium transition-colors duration-200"
                    style={{
                      backgroundColor,
                    }}
                  >
                    Sentiment Score: {entity[2].toFixed(2)}
                  </span>
                </div>

                {entity[3] && (
                  <div className="space-y-2">
                    <h5 className="font-semibold text-gray-700 text-sm">
                      Key Points
                    </h5>
                    <div className="text-sm text-gray-600">
                      <SummarizedSentimentBulletPoints summary={entity[3]} />
                    </div>
                  </div>
                )}
                <LinksForEntity entity={entity} />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

const NamedEntitiesMenu: React.FC<{
  dates: string[];
  currentDate: string;
  setCurrentDate: (date: string) => void;
  timeFilter: string;
}> = ({ dates, currentDate, setCurrentDate, timeFilter }) => {
  if (window.innerWidth < 500) {
    return (
      <div className="pb-4">
        <label className="block text-sm text-center font-medium text-gray-700">
          Date
        </label>
        <select
          value={currentDate}
          onChange={(e) => {
            setCurrentDate(e.target.value);
          }}
          className="mt-1 block w-full px-1 py-2 text-base border border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
        >
          {dates.map((date) => (
            <option key={date} value={date}>
              {formatDate(date, timeFilter)}
            </option>
          ))}
        </select>
      </div>
    );
  }

  return (
    <div className="flex mt-2 mb-6 w-auto bg-white border border-gray-300 z-10 rounded-md">
      {dates.map((date, idx) => (
        <div
          key={date}
          className={idx !== 0 ? "border-l border-gray-300" : ""}
        >
          <div
            onClick={() => setCurrentDate(date)}
            className={`px-4 py-2 cursor-pointer hover:bg-indigo-50 ${
              currentDate === date ? "bg-indigo-100 text-black" : ""
            }`}
          >
            {formatDate(date, timeFilter)}
          </div>
        </div>
      ))}
    </div>
  );
};

export const NamedEntities: React.FC<NamedEntitiesProps> = ({
  analysis,
  timeFilter,
}) => {
  const [currentNamedEntityDate, setCurrentNamedEntityDate] = useState(
    Object.keys(analysis.top_named_entities)[0]
  );

  return (
    <div className="mt-6">
      <h2
        className="text-xl font-bold text-center"
        style={{ textAlign: "center", fontSize: "20px" }}
      >
        Most Mentioned Named Entities
      </h2>
      <div className="flex relative items-center text-center justify-center">
        <br />
        <ColorKey />
      </div>
      <div className="flex flex-col items-center justify-center">
        <NamedEntitiesMenu
          dates={Array.from(Object.keys(analysis.top_named_entities))}
          currentDate={currentNamedEntityDate}
          setCurrentDate={setCurrentNamedEntityDate}
          timeFilter={timeFilter}
        />
        <TopNamedEntitiesForDate
          date={currentNamedEntityDate}
          entities={analysis.top_named_entities[currentNamedEntityDate]}
          timeFilter={timeFilter}
        />
      </div>
      <div className="flex flex-col md:flex-row gap-5 mb-5 mt-5 items-center justify-center">
        <ButtonPopover title="What's a named entity?">
          A Named Entity is a key subject in a piece of text (include names of
          people, organizations, locations, and dates)
        </ButtonPopover>
        <ButtonPopover title="How's sentiment score computed?">
          The aspect-based sentiment analysis model
          <strong>"yangheng/deberta-v3-base-absa-v1.1"</strong>
          is used to generate a sentiment score from -1 (most negative) to 1
          (most positive). This sentiment score represents how the writer feels
          TOWARD the named entity (and is independent of the sentence's overall
          tone & sentiment).
          <br />
          <hr />
          <br />
          Example:
          <i>
            "It's so unfair that Taylor Swift didn't get nominated for a grammy
            this year - she got cheated"
          </i>
          The overall tone/sentiment of this sentence is frustrated/angry, but
          the writer feels{" "}
          <strong style={{ color: "green" }}> POSITIVELY </strong>
          toward Taylor Swift because they believe she is deserving of a grammy.
        </ButtonPopover>
        <ButtonPopover title='How are "Key Points" computed?'>
          The summarization ML model "facebook/bart-large-cnn" processes all of
          the sentences where a named entity is directly mentioned and creates a
          short summary.
        </ButtonPopover>
      </div>
    </div>
  );
};
