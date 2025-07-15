import { SubredditAnalysis } from "../../types/redditAnalysis";
import ButtonPopover from "../ButtonPopover";
import { EntityPicture } from "./EntityPicture";
import { entity_name_to_png } from "../../constants/named_entity_png.ts";
import { useRef, useEffect, useState } from 'react';

interface NamedEntitiesProps {
  analysis: SubredditAnalysis;
  timeFilter: string;
}

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
  analysis
}) => {
  const [popupIdx, setpopupIdx] = useState(-1)
  const handleClick = (entity_idx: number) => { setpopupIdx(entity_idx); };
  const closePopup = () => { 
    console.log('hi')
    setpopupIdx(-1); 
    console.log("popupIdx: ", popupIdx)
  };

  let top_named_entities_with_picture = []
  for (let i = 0; i < analysis.top_named_entities.length; i++) {
    const entity = analysis.top_named_entities[i]
    if (entity_name_has_picture(entity.name)) {
      top_named_entities_with_picture.push(entity)
    }
  }

  const popupRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (popupIdx !== -1 && popupRef.current) {
      const popup = popupRef.current;
      const rect = popup.getBoundingClientRect();
      const buffer = 16; // pixels from screen edge

      const overflowRight = rect.right > window.innerWidth - buffer;
      const overflowLeft = rect.left < buffer;

      if (overflowRight) {
        popup.style.left = `-${rect.right - window.innerWidth + buffer}px`;
      } else if (overflowLeft) {
        popup.style.left = `${-rect.left + buffer}px`;
      } else {
        popup.style.left = "0px";
      }
    }
  }, [popupIdx]);

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
          {analysis.top_named_entities.map((named_entity, entity_idx) => (
              <div className="font-semibold relative inline-block">
                <div onClick={() => handleClick(entity_idx)}>
                  <EntityPicture entity_name={named_entity.name}/>
                </div>
                {/* Inline popup (shows below the div) */}
                {popupIdx != -1 && popupIdx === entity_idx && (
                  <div
                    ref={popupRef}
                    className="absolute mt-2 w-150 bg-gray-600 text-gray-200 border border-gray-300 rounded-md shadow-lg z-10"
                  >
                     {/* Red X button in the top-right corner */}
                      <button
                        onClick={closePopup} 
                        className="absolute bg-red-400 top-1 right-1 text-white pl-2 pr-2 hover:text-gray-500 text-xl font-bold"
                      >
                        ×
                      </button>
                      <h1 className="rounded-md text-center pt-2 pb-2 bg-gray-500 text-gray-100">AI analysis based on comments discussing "{named_entity.name}"</h1>
                      <p className="p-5 text-sm font-normal leading-relaxed">{named_entity.AI_analysis}</p>
                  </div>
                )}
              </div>
          ))}
        </div>
        </div>
    </div>
  );
};
