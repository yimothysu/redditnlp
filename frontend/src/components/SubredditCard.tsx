import { SubredditAvatar, DefaultSubredditAvatar } from "./SubredditAvatar";
import { hashColor } from "../lib/hashColor";

interface SubredditCardProps {
  subredditName: string;
}

export function SubredditCard({ subredditName }: SubredditCardProps) {
  if (subredditName == "AskOldPeople" || subredditName == "personalfinance") {
    return (
      <a
      href={`/subreddit/${subredditName}`}
      className="bg-white outline outline-1 rounded-md p-4 w-35 h-35 flex justify-center flex-col items-center text-center shadow hover:cursor-pointer text-black! text-[12.5px]"
    >
      <div className="w-20 h-20 mb-4 flex justify-center items-center">
        <DefaultSubredditAvatar/>
      </div>
      <b>r/{subredditName}</b>
    </a>
    ); 
  }
  return (
    <a
      href={`/subreddit/${subredditName}`}
      className="bg-white outline outline-1 rounded-md p-4 w-35 h-35 flex justify-center flex-col items-center text-center shadow hover:cursor-pointer text-black! text-[12.5px]"
    >
      <div className="w-20 h-20 mb-4 flex justify-center items-center">
        <SubredditAvatar subredditName={subredditName} />
      </div>
      <b>r/{subredditName}</b>
    </a>
  );
}
