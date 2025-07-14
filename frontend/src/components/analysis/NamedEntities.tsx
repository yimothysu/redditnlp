import { useState } from "react";
import { SubredditAnalysis, NamedEntity } from "../../types/redditAnalysis";
import ButtonPopover from "../ButtonPopover";
import { EntityPicture } from "./EntityPicture";
import { entity_name_to_png } from "../../constants/named_entity_png.ts";

interface NamedEntitiesProps {
  analysis: SubredditAnalysis;
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

// const SummarizedSentimentBulletPoints: React.FC<{
//   entity_name: string, date: string, num_comments_summarized: number, key_points: string;
// }> = ({ entity_name, date, num_comments_summarized, key_points }) => {
//   if (!key_points) return null;
//   const trimmed_key_points = key_points.replace("Here is the summary:", "");
//   if(trimmed_key_points != null) {
//     const sentences = (trimmed_key_points.match(/[^.!?]+[.!?]+[\])'"`’”]*|\s*$/g) ?? []).map(s => s.trim());
//     const cleaned_sentences = sentences.filter(sentence => sentence.trim() !== "");    
//     if(cleaned_sentences.length == 0) {
//       console.log(key_points)
//       return; 
//     }
//     const sentiment_values = new Map([
//       ["very positive", "very_positive.png"],
//       ["positive leaning", "positive_leaning.png"],
//       ["neutral", "neutral.png"],
//       ["mixed", "neutral.png"],
//       ["very negative", "very_negative.png"],
//       ["negative leaning", "negative_leaning.png"]
//     ]);
    
//     const firstSentence = cleaned_sentences[0].toLowerCase(); 
//     const sentiment_value = Array.from(sentiment_values.keys()).find(sentiment_value =>
//       firstSentence.includes(sentiment_value));

//     if(sentiment_value != null) {
//       const outline_color =
//       sentiment_value == "very positive" ? "#91CC91" :
//       sentiment_value == "positive leaning" ? "#d3ea84" :
//       sentiment_value == "neutral" ? "#ffffac" :
//       sentiment_value == "mixed" ? "#ffffac" :
//       sentiment_value == "negative leaning" ? "#ffc245" :
//       sentiment_value == "very negative" ? "#ff9898" :
//       "bg-gray-100";

//       const background_color = 
//       outline_color == "#91CC91" ? Color(outline_color).lighten(0.35).hex() :
//       outline_color == "#d3ea84" ? Color(outline_color).lighten(0.3).hex() :
//       outline_color == "#ffffac" ? Color(outline_color).lighten(0.15).hex() :
//       outline_color == "#ffc245" ? Color(outline_color).lighten(0.45).hex() :
//       outline_color == "#ff9898" ? Color(outline_color).lighten(0.2).hex() : 
//       "bg-gray-100";

//       return (
//         <div>
//           <div className="border border-3 p-0.5 border-gray-100 rounded-md mb-7">
//             <div className="flex flex-col shadow rounded-sm"
//             style={{ outline: `2px solid ${outline_color}`, backgroundColor: `${background_color}`}}>
//               <div className="flex flex-row gap-3 pb-2 items-center justify-center font-semibold pt-2">
//                 <div className="text-[14.5px] mt-1 text-center text-black">This subreddit's opinion of <span className="font-bold">{entity_name}</span> on {date}: {sentiment_value}</div>
//                 <img
//                   src={"/moods/" + sentiment_values.get(sentiment_value)}
//                   className="w-7 object-cover cursor-pointer transition active:scale-95 active:brightness-90"
//                 />
//               </div>
//             </div>
//           </div>
//           <h1 className="mb-2 text-[12.5px] text-center italic text-gray-500">(Summary of <span className="font-semibold text-red-600">{num_comments_summarized}</span> comments regarding {entity_name})</h1>
//           {cleaned_sentences.slice(1).map((sentence, index) => (
//             <li className="text-[14px] pb-3 text-gray-600 leading-loose" key={index}>{sentence}</li>
//           ))}
//         </div>
//       );
//     }
//   }
// };

// const LinksForEntity = ({ entity }: { entity: NamedEntity }) => {
//   if (Object.keys(entity.urls).length === 0) {
//     return null;
//   }

//   console.log("entity[4].length: ", entity.urls.length);
//   if (Number(entity.urls.length) == 1) {
//     return (
//       <div className="">
//         {Object.entries(entity.urls).map(([_, link]) => (
//           <div
//             onClick={() => window.open(link, "_blank")}
//             className="w-20 bg-[#fa6f4d] hover:bg-gray-100 transition-colors duration-200 p-2 text-sm font-medium rounded-lg text-white text-center flex items-center justify-center cursor-pointer border border-gray-100"
//           >
//             post
//             <ChevronRight className="w-5 h-5 ml-2 text-white" />
//           </div>
//         ))}
//       </div>
//     );
//   } else {
//     return (
//       <div className="">
//         {Object.entries(entity.urls).map(([_, link]) => (
//           <div
//             onClick={() => window.open(link, "_blank")}
//             className="w-20 m-1 bg-[#fa6f4d] hover:bg-gray-100 transition-colors duration-200 p-2 text-sm font-medium rounded-lg text-white text-center flex items-center justify-center cursor-pointer border border-gray-100"
//           >
//             post
//             <ChevronRight className="w-5 h-5 ml-2 text-white" />
//           </div>
//         ))}
//       </div>
//     );
//   }
// };

// const TopNamedEntitiesForDate: React.FC<TopNamedEntitiesForDateProps> = ({
//   date,
//   entities,
// }) => {
//   const label_to_entities = GroupEntitiesByLabel({entities})

//   const EntityLabel = ({label}: {label: string}) => {
//     let displayed_label = ""
//     if(label === "PERSON") {
//       displayed_label = "People"
//     }
//     else if(label === "ORG") {
//       displayed_label = "Organizations"
//     }
//     else if(label === "GPE") {
//       displayed_label = "Geopolitical Locations"
//     }
//     else if(label === "LOC") {
//       displayed_label = "Non-Geopolitical Locations"
//     }
//     else if(label === "FAC") {
//       displayed_label = "Facilities"
//     }
//     else if(label === "PRODUCT") {
//       displayed_label = "Products"
//     }
//     else if(label === "WORK_OF_ART") {
//       displayed_label = "Creative Works"
//     }
//     else if(label === "LAW") {
//       displayed_label = "Named documents or legal rules"
//     }
//     else if(label === "EVENT") {
//       displayed_label = "Named events"
//     }
//     else if(label === "LANGUAGE") {
//       displayed_label = "Named Languages"
//     }
//     else if(label === "NORP") {
//       displayed_label = "Nationalities, religious and political groups"
//     }
//     return (
//       <h1 className="bg-indigo-100 text-[18px] font-semibold text-center p-2">{displayed_label}</h1>
//     )
//   };

//   return (
//     <div
//       key={date}
//       className="flex-grow flex-1 mb-6 bg-white overflow-hidden"
//       >
//       {Object.entries(label_to_entities).map(([label, entities_for_label]) => (
//       <>
//         <EntityLabel label={label}></EntityLabel>
//         <div className="grid grid-cols-1 gap-0.5 bg-gray-100 mb-10">
//           {entities_for_label.map((entity: NamedEntity, index: number) => {
//             return (
//               <div
//                 key={index}
//                 className="pt-1 pb-1 grid md:[grid-template-columns:200px_800px_100px] bg-white transition-all duration-200"
//               >
//                   {/* Column 1 */}
//                   <div className="border-r border-gray-200 last:border-r-0 flex flex-col p-1 gap-3 justify-center items-center">
//                     <h4 className="mt-3 text-[17px] mb-3 font-bold text-gray-600">{entity.name}</h4>
//                     <EntityPicture entity_name={entity.name}></EntityPicture>
//                   </div>
//                   {/* Column 2 */}
//                   <div className="p-5 border-r border-gray-200 last:border-r-0 p-2 bg-gray-50 text-left">
//                     {entity.key_points && (
//                       <div className="">
//                         <div className="text-[12px] text-gray-500">
//                           <SummarizedSentimentBulletPoints entity_name={entity.name} date={date} num_comments_summarized={entity.num_comments_summarized} key_points={entity.key_points} />

//                         </div>
//                       </div>
//                     )}
//                   </div>
//                   {/* Column 4 */}
//                   <div className="p-2 bg-white">
//                     <LinksForEntity entity={entity} /> 
//                   </div>
//                 </div>
//             );
//           })}
//         </div>
//         </>
//       ))}
//       </div>
//   );
// };

function entity_name_has_picture(entity_name: string) {
  entity_name = entity_name.toLowerCase();
  let picture_file_name = null
  if(entity_name in entity_name_to_png) {
    picture_file_name = entity_name_to_png[entity_name];
  }
  else if (entity_name.startsWith("the ") && entity_name.slice(4) in entity_name_to_png) {
    picture_file_name = entity_name_to_png[entity_name.slice(4)];
  }
  else if (entity_name.startsWith("an ") && entity_name.slice(3) in entity_name_to_png) {
    picture_file_name = entity_name_to_png[entity_name.slice(3)];
  }
  else if (!entity_name.endsWith("s") && (entity_name + "s") in entity_name_to_png) {
    picture_file_name = entity_name_to_png[entity_name + "s"];
  }
  else if (entity_name.endsWith("s") && (entity_name.slice(0, -1)) in entity_name_to_png) {
    picture_file_name = entity_name_to_png[entity_name.slice(0, -1)];
  }

  if (picture_file_name != null) {
    return true
  }
  return false 
};

export const NamedEntities: React.FC<NamedEntitiesProps> = ({
  analysis,
  timeFilter,
}) => {

  let top_named_entities_with_picture = []
  for (let i = 0; i < analysis.top_named_entities.length; i++) {
    const entity = analysis.top_named_entities[i]
    if (entity_name_has_picture(entity.name)) {
      top_named_entities_with_picture.push(entity)
    }
  }

  return (
    <div className="mt-6">
      <h2
        className="text-xl font-bold text-center"
        style={{ textAlign: "center", fontSize: "20px" }}
      >
        Named Entities
      </h2>
      <div className="flex flex-col mt-2">
        <div className="flex flex-col md:flex-row gap-5 mb-8 mt-4 items-center justify-center">
          <ButtonPopover title="What's a named entity?">
            A Named Entity is a key subject in a piece of text (include names of
            people, organizations, locations, and dates)
          </ButtonPopover>
        </div>
        <div className="grid grid-cols-5 mb-4 gap-y-10">
          {analysis.top_named_entities.map((named_entity, index) => (
              <div className="font-semibold">
                <EntityPicture entity_name={named_entity.name} />
              </div>
          ))}
        </div>
        </div>
    </div>
  );
};
