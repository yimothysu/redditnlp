//import { hashColor } from "../lib/hashColor";

interface SubredditAvatarProps {
  subredditName: string;
}

export function SubredditAvatar({ subredditName }: SubredditAvatarProps) {
  // const color = hashColor(subredditName);

  // return (
  //   <div
  //     className="flex items-center justify-center rounded-full w-20 h-20 font-bold text-white"
  //     style={{ backgroundColor: color }}
  //   >
  //     {subredditName[0].toUpperCase()}
  //   </div>
  // );
  const image_src = "/subreddit_pics/" + subredditName + ".png"
  return (
    <img 
      src={image_src}
      className="rounded-full w-15 h-15 object-cover"
    />
  );
}

export function DefaultSubredditAvatar() {
  const color = '#4268e8'

  return (
    <div
      className="flex items-center justify-center rounded-full w-15 h-15 font-semibold text-[30px]"
      style={{ backgroundColor: color }}
    >
      r/
    </div>
  );
}