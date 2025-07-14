import { SubredditAnalysis } from "../../types/redditAnalysis";
import ButtonPopover from "../ButtonPopover";
import { EntityPicture } from "./EntityPicture";
import { entity_name_to_png } from "../../constants/named_entity_png.ts";

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
          {analysis.top_named_entities.map((named_entity, _) => (
              <div className="font-semibold">
                <EntityPicture entity_name={named_entity.name} />
              </div>
          ))}
        </div>
        </div>
    </div>
  );
};
