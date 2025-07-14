import { entity_name_to_png } from "../../constants/named_entity_png.ts";
 
export const EntityPicture = ({entity_name}: { entity_name: string }) => {
    let picture_file_name = null;
    if(entity_name in entity_name_to_png) {
      picture_file_name = entity_name_to_png[entity_name];
    }
    else {
      entity_name = entity_name.toLowerCase();
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
    }

    if (picture_file_name != null) {
      return (
        <div className="flex flex-col rounded-sm m-1 shadow-sm items-center w-50 h-50">
          <h1 className="font-semibold text-gray-500 bg-gray-100 w-50 text-center text-md p-1 mb-3">{entity_name[0].toUpperCase() + entity_name.slice(1)}</h1>
          <img
            src={"/named_entity_pics/" + picture_file_name}
            className="w-30 object-cover cursor-pointer transition active:scale-95 active:brightness-90"
          />
        </div>
      );
    }
    else {
      return (
        <div className="flex flex-col rounded-sm p-1 shadow-sm items-center w-50 h-50">
          <h1 className="font-semibold text-gray-500 bg-gray-100 w-50 p-1 text-center text-md mb-3">{entity_name[0].toUpperCase() + entity_name.slice(1)}</h1>
          <img
            src={"/no_image.png"}
            className="w-30 object-cover cursor-pointer transition active:scale-95 active:brightness-90"
          />
        </div>
      )
    }
  };