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

function GroupEntitiesByLabel({entities}: { entities: NamedEntity[] }) {
  const label_to_entities: Record<string, NamedEntity[]> = {};
  for(const entity of entities) {
    if (!label_to_entities[entity.label]) {
      label_to_entities[entity.label] = [];
    }
    label_to_entities[entity.label].push(entity);
  }
  return label_to_entities 
}

const SummarizedSentimentBulletPoints: React.FC<{
  entity_name: string, date: string, num_comments_summarized: string, key_points: string;
}> = ({ entity_name, date, num_comments_summarized, key_points }) => {
  if (!key_points) return null;
  const trimmed_key_points = key_points.replace("Here is the summary:", "");
  if(trimmed_key_points != null) {
    const sentences = trimmed_key_points.match(/[^.!?]+[.!?]+[\])'"`’”]*|\s*$/g).map(s => s.trim());
    const cleaned_sentences = sentences.filter(sentence => sentence.trim() !== "");    
    const sentiment_values = new Map([
      ["very positive", "very_positive.png"],
      ["positive leaning", "positive_leaning.png"],
      ["neutral", "neutral.png"],
      ["mixed", "neutral.png"],
      ["very negative", "very_negative.png"],
      ["negative leaning", "negative_leaning.png"]
    ]);
    
    const firstSentence = cleaned_sentences[0].toLowerCase(); 
    const sentiment_value = Array.from(sentiment_values.keys()).find(sentiment_value =>
      firstSentence.includes(sentiment_value));

    if(sentiment_value != null) {
      const outline_color =
      sentiment_value == "very positive" ? "#91CC91" :
      sentiment_value == "positive leaning" ? "#d3ea84" :
      sentiment_value == "neutral" ? "#ffffac" :
      sentiment_value == "mixed" ? "#ffffac" :
      sentiment_value == "negative leaning" ? "#ffc245" :
      sentiment_value == "very negative" ? "#ff9898" :
      "bg-gray-100";
      return (
        <div>
          <div className="border border-3 p-0.5 border-gray-100 rounded-md mb-5">
            <div className="flex flex-col shadow rounded-sm bg-white"
            style={{ outline: `2px solid ${outline_color}`}}>
              <div className="flex flex-row gap-3 pb-3 items-center justify-center font-semibold pt-3">
                <div className="text-[14.5px] mt-3 text-center text-black">This subreddit's overall opinion towards {entity_name} on {date}: {sentiment_value}</div>
                <img
                  src={"/moods/" + sentiment_values.get(sentiment_value)}
                  className="w-9 object-cover cursor-pointer transition active:scale-95 active:brightness-90"
                />
              </div>
              <h1 className="mb-4 text-[13px] pb-2 text-center font-semibold pt-2 text-black">(Summary of <span className="font-bold text-red-600">{num_comments_summarized}</span> comments regarding {entity_name})</h1>
            </div>
          </div>
          {cleaned_sentences.slice(1).map((sentence, index) => (
            <li className="text-[14px] pb-3 leading-loose" key={index}>{sentence}</li>
          ))}
        </div>
      );
    }
  }
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
      <div className="">
        {Object.entries(entity.urls).map(([_, link]) => (
          <div
            onClick={() => window.open(link, "_blank")}
            className="w-20 m-1 bg-[#fa6f4d] hover:bg-gray-100 transition-colors duration-200 p-2 text-sm font-medium rounded-lg text-white text-center flex items-center justify-center cursor-pointer border border-gray-100"
          >
            post
            <ChevronRight className="w-5 h-5 ml-2 text-white" />
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
  const label_to_entities = GroupEntitiesByLabel({entities})

  const EntityLabel = ({label}: {label: string}) => {
    let displayed_label = ""
    if(label === "PERSON") {
      displayed_label = "People"
    }
    else if(label === "ORG") {
      displayed_label = "Organizations"
    }
    else if(label === "GPE") {
      displayed_label = "Geopolitical Locations"
    }
    else if(label === "LOC") {
      displayed_label = "Non-Geopolitical Locations"
    }
    else if(label === "FAC") {
      displayed_label = "Facilities"
    }
    else if(label === "PRODUCT") {
      displayed_label = "Products"
    }
    else if(label === "WORK_OF_ART") {
      displayed_label = "Creative Works"
    }
    else if(label === "LAW") {
      displayed_label = "Named documents or legal rules"
    }
    else if(label === "EVENT") {
      displayed_label = "Named events"
    }
    else if(label === "LANGUAGE") {
      displayed_label = "Named Languages"
    }
    else if(label === "NORP") {
      displayed_label = "Nationalities, religious and political groups"
    }
    return (
      <h1 className="bg-indigo-100 text-[18px] font-semibold text-center p-2">{displayed_label}</h1>
    )
  };

  return (
    <div
      key={date}
      className="flex-grow flex-1 mb-6 bg-white overflow-hidden"
      >
      {Object.entries(label_to_entities).map(([label, entities_for_label]) => (
      <>
        <EntityLabel label={label}></EntityLabel>
        <div className="grid grid-cols-1 gap-0.5 bg-gray-100">
          {entities_for_label.map((entity: NamedEntity, index: number) => {
            return (
              <div
                key={index}
                className="pt-1 pb-1 grid md:[grid-template-columns:200px_800px_100px] bg-white transition-all duration-200"
              >
                  {/* Column 1 */}
                  <div className="border-r border-gray-200 last:border-r-0 flex p-1 gap-3 justify-center items-center">
                    <h4 className="mt-3 text-[15px] font-semibold text-gray-800">{entity.name}</h4>
                    <EntityPicture entity_name={entity.name}></EntityPicture>
                  </div>
                  {/* Column 2 */}
                  <div className="p-5 border-r border-gray-200 last:border-r-0 p-2 bg-gray-50 text-left">
                    {entity.key_points && (
                      <div className="">
                        <div className="text-[12px] text-gray-500">
                          <SummarizedSentimentBulletPoints entity_name={entity.name} date={date} num_comments_summarized={entity.num_comments_summarized} key_points={entity.key_points} />

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
        </>
      ))}
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
        {/* <div className="w-full">
          <ColorKey />
        </div> */}
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
