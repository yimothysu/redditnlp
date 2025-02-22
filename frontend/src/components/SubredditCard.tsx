import { hashColor } from "../lib/hashColor";

interface SubredditCardProps {
  subredditName: string;
}

export function SubredditCard({ subredditName }: SubredditCardProps) {
  const color = hashColor(subredditName);

  return (
    <a
      href={`/subreddit/${subredditName}`}
      className="bg-white rounded-md p-10 text-center shadow hover:cursor-pointer text-black!"
    >
      <div
        className="flex items-center justify-center rounded-full w-16 h-16 mb-4 font-bold"
        style={{ backgroundColor: color }}
      >
        {subredditName[0].toUpperCase()}
      </div>
      <b>r/{subredditName}</b>
    </a>
  );
}
