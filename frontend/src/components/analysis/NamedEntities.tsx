import { useState } from "react";
import { SubredditAnalysis, NamedEntity } from "../../types/redditAnalysis";
import ButtonPopover from "../ButtonPopover";
import { ColorKey } from "./ColorKey";
import { formatDate } from "../../utils/dateUtils";
import { ChevronRight } from "lucide-react";
import { EntityPicture } from "./EntityPicture";

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
  key_points: string[];
}> = ({ key_points }) => {
  if (!key_points) return null;
  return (
    <ul className="list-disc pl-4 max-w-[450px]">
      {key_points.map((sentence: string, index: number) => (
        <li className="pb-1" key={index}>{sentence.trim()}</li>
      ))}
    </ul>
  );
};

const LinksForEntity = ({ entity }: { entity: NamedEntity }) => {
  if (Object.keys(entity.urls).length === 0) {
    return null;
  }

  console.log("entity[4].length: ", entity.urls.length);
  if (Number(entity.urls.length) == 1) {
    return (
      <div className="">
        {Object.entries(entity.urls).map(([_, link]) => (
          <div
            onClick={() => window.open(link, "_blank")}
            className="w-20 bg-[#fa6f4d] hover:bg-gray-100 transition-colors duration-200 p-2 text-sm font-medium rounded-lg text-white text-center flex items-center justify-center cursor-pointer border border-gray-100"
          >
            post
            <ChevronRight className="w-5 h-5 ml-2 text-white" />
          </div>
        ))}
      </div>
    );
  } else {
    return (
      <div className="mt-4">
        {Object.entries(entity.urls).map(([idx, link]) => (
          <div
            onClick={() => window.open(link, "_blank")}
            className="m-3 bg-gray-50 hover:bg-gray-100 transition-colors duration-200 p-3 text-sm font-medium rounded-lg text-gray-700 text-center flex items-center justify-center cursor-pointer border border-gray-100"
          >
            #{Number(idx) + 1} Top Post discussing "{entity.name}"
            <ChevronRight className="w-5 h-5 ml-2 text-gray-500" />
          </div>
        ))}
      </div>
    );
  }
};

const TopNamedEntitiesForDate: React.FC<TopNamedEntitiesForDateProps> = ({
  date,
  entities,
}) => {

  return (
    <div
      key={date}
      className="flex-grow flex-1 mb-6 bg-white overflow-hidden"
    >
      <div className="grid grid-cols-1 gap-0.5 bg-gray-100">
        {entities.map((entity: NamedEntity, index: number) => {
          const backgroundColor =
            entity.sentiment >= 0.6 && entity.sentiment <= 1
              ? "#91CC91"
              : entity.sentiment >= 0.2 && entity.sentiment < 0.6
              ? "#d3ea84"
              : entity.sentiment >= -0.2 && entity.sentiment < 0.2
              ? "#ffffac"
              : entity.sentiment >= -0.6 && entity.sentiment < -0.2
              ? "#ffc245"
              : entity.sentiment >= -1 && entity.sentiment < -0.6
              ? "#ff9898"
              : "bg-gray-100";
          return (
            <div
              key={index}
              className="pt-1 pb-1 grid [grid-template-columns:200px_350px_450px_100px] bg-white transition-all duration-200"
            >
                {/* Column 1 */}
                <div className="border-r border-gray-200 last:border-r-0 flex p-1 bg-gray-50 gap-3 justify-center items-center">
                  <h4 className="mt-3 text-[15px] font-bold text-gray-800">{entity.name}</h4>
                  <EntityPicture entity_name={entity.name}></EntityPicture>
                </div>
                {/* Column 2 */}
                <div className="border-r border-gray-200 last:border-r-0  flex gap-2 justify-start items-center">
                  <div
                    className="h-6 transition-colors duration-200"
                    style={{
                      backgroundColor: backgroundColor,
                      width: `${(((entity.sentiment + 1) / 2) * 350)}px`
                    }}
                  >
                  </div>
                  <div className="text-gray-700 text-xs font-semibold italic">
                    {entity.sentiment.toFixed(2)}
                  </div>
                </div>
                {/* Column 3  */}
                <div className="border-r border-gray-200 last:border-r-0  p-2 bg-gray-50 text-left">
                  {entity.key_points && (
                    <div className="">
                      <div className="text-[12px] text-gray-500">
                        <SummarizedSentimentBulletPoints key_points={entity.key_points} />

                      </div>
                    </div>
                  )}
                </div>
                                {/* Column 4 */}
                                <div className="p-2 bg-white">
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
        <div key={date} className={idx !== 0 ? "border-l border-gray-300" : ""}>
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
    Object.keys(analysis.top_named_entities).sort(
      (a, b) => new Date(b).getTime() - new Date(a).getTime()
    )[0]
  );

  return (
    <div className="mt-6">
      <h2
        className="text-xl font-bold text-center"
        style={{ textAlign: "center", fontSize: "20px" }}
      >
        Named Entities
      </h2>
      <div className="mt-3 flex flex-col items-center justify-center">
        <NamedEntitiesMenu
          dates={Array.from(
            Object.keys(analysis.top_named_entities).sort(
              (a, b) => new Date(b).getTime() - new Date(a).getTime()
            )
          )}
          currentDate={currentNamedEntityDate}
          setCurrentDate={setCurrentNamedEntityDate}
          timeFilter={timeFilter}
        />
      </div>
      <div className="flex flex-col mt-5">
        <div className="w-full">
          <ColorKey />
        </div>
        <hr className="border-t-2 border-gray-400"></hr>
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
