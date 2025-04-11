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
    <div className="mt-7">
      {Object.entries(entity[4]).map(([_, link]) => (
        <div
          onClick={() => window.open(link, "_blank")}
          className="bg-gray-100 hover:bg-gray-200 active:bg-gray-300 p-2 text-[14px] font-semibold items-center justify-center shadow rounded text-black text-center flex"
        >
          Top Post discussing "{entity[0]}"
          <ChevronRight className="w-6 h-6 ml-5 text-black" />
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
      className="flex-grow flex-1 mb-6 mr-1 ml-1 border bg-white border-gray-300 rounded-l shadow-md"
    >
      <h3
        className="text-lg font-semibold bg-gray-100"
        style={{
          fontSize: "20px",
          textAlign: "center",
          padding: "4px",
        }}
      >
        {formatDate(date, timeFilter)}
      </h3>
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-3 pl-0">
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
              className="border border-gray-300 p-4 shadow-sm bg-white"
            >
              <div className="flex justify-center">
                <div className="inline-block justify-center items-center text-center font-[18px] font-bold pl-3 pr-3 pt-1 pb-1">
                  {entity[0]}
                </div>
              </div>
              <div className="ml-5 mr-5 mt-3 mb-3">
                <div className="flex flex-col items-center">
                  <span
                    className="pl-3 pr-3 pt-1 pb-1"
                    style={{
                      fontWeight: 600,
                      fontSize: "15px",
                      backgroundColor: backgroundColor,
                      width: "fit-content",
                    }}
                  >
                    Sentiment Score {"  "} = {"  "}
                    {entity[2]}
                  </span>
                </div>

                {entity[3] != "" && (
                  <div
                    style={{
                      fontWeight: 600,
                      fontSize: "14px",
                      marginTop: "20px",
                      marginBottom: "5px",
                    }}
                  >
                    Key Points
                  </div>
                )}
                <div
                  style={{
                    fontSize: "13.5px",
                    color: "#333",
                  }}
                >
                  <SummarizedSentimentBulletPoints summary={entity[3]} />
                  <LinksForEntity entity={entity} />
                </div>
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
    <div className="flex mt-2 mb-6 w-auto bg-white border border-2 border-gray-300 shadow-lg z-10">
      {dates.map((date, idx) => (
        <div
          key={date}
          className={idx !== 0 ? "border-l-2 border-gray-300" : ""}
        >
          <div
            onClick={() => setCurrentDate(date)}
            className={`px-4 py-2 cursor-pointer ${
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
