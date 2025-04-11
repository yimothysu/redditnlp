import { SubredditAvatar, DefaultSubredditAvatar } from "./SubredditAvatar";

interface SubredditCardProps {
  subredditName: string;
}

export function SubredditCard({ subredditName }: SubredditCardProps) {
  if (subredditName == "AskOldPeople" || subredditName == "personalfinance") {
    return (
      <a
        href={`/subreddit/${subredditName}`}
        className="mb-2 md:mb-0 bg-white outline rounded-md p-2 w-27 h-25 md:w-35 md:h-35 flex justify-center flex-col items-center text-center shadow hover:cursor-pointer text-black! text-[10px] md:text-[12.5px]"
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
      className="mb-2 md:mb-0 bg-white outline rounded-md p-2 w-27 h-25 md:w-35 md:h-35 flex justify-center flex-col items-center text-center shadow hover:cursor-pointer text-black! text-[10px] md:text-[12.5px]"
    >
      <div className="mb-4 flex justify-center items-center">
        <SubredditAvatar subredditName={subredditName} />
      </div>
      <b>r/{subredditName}</b>
    </a>
  );
}
