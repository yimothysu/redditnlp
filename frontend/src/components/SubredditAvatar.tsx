import { hashColor } from "../lib/hashColor";

interface SubredditAvatarProps {
  subredditName: string;
}

export function SubredditAvatar({ subredditName }: SubredditAvatarProps) {
  const color = hashColor(subredditName);

  return (
    <div
      className="flex items-center justify-center rounded-full w-20 h-20 font-bold text-white"
      style={{ backgroundColor: color }}
    >
      {subredditName[0].toUpperCase()}
    </div>
  );
}
