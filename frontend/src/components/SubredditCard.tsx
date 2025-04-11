import { SubredditAvatar, DefaultSubredditAvatar } from "./SubredditAvatar";

interface SubredditCardProps {
  subredditName: string;
}

export function SubredditCard({ subredditName }: SubredditCardProps) {
  if (subredditName == "AskOldPeople" || subredditName == "personalfinance") {
    return (
      <a
        href={`/subreddit/${subredditName}`}
        className="mb-2 md:mb-0 bg-white rounded-md p-2 w-40 h-40 flex justify-center flex-col items-center text-center hover:cursor-pointer text-black! text-[10px] md:text-[12.5px] border border-gray-300 hover:border-gray-400"
      >
        <div className="mb-4 flex justify-center items-center">
          <DefaultSubredditAvatar />
        </div>
        <b>r/{subredditName}</b>
      </a>
    );
  }
  return (
    <a
      href={`/subreddit/${subredditName}`}
      className="mb-2 md:mb-0 bg-white rounded-md p-2 w-40 h-40 flex justify-center flex-col items-center text-center hover:cursor-pointer text-black! text-[10px] md:text-[12.5px] border border-gray-300 hover:border-gray-400"
    >
      <div className="mb-4 flex justify-center items-center">
        <SubredditAvatar subredditName={subredditName} />
      </div>
      <b>r/{subredditName}</b>
    </a>
  );
}
