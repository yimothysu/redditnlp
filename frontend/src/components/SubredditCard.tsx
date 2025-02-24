import { hashColor } from "../lib/hashColor";
import { SubredditAvatar } from "./SubredditAvatar";

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
      <SubredditAvatar subredditName={subredditName} />
      <b>r/{subredditName}</b>
    </a>
  );
}
