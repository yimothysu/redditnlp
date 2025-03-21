import { SubredditAvatar } from "./SubredditAvatar";

interface SubredditCardProps {
  subredditName: string;
}

export function SubredditCard({ subredditName }: SubredditCardProps) {
  return (
    <a
      href={`/subreddit/${subredditName}`}
      className="bg-white rounded-md p-6 w-64 h-64 flex justify-center flex-col items-center text-center shadow hover:cursor-pointer text-black!"
    >
      <div className="w-20 h-20 mb-4">
        <SubredditAvatar subredditName={subredditName} />
      </div>
      <b>r/{subredditName}</b>
    </a>
  );
}
