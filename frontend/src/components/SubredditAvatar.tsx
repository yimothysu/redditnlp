interface SubredditAvatarProps {
  subredditName: string;
}

export function SubredditAvatar({ subredditName }: SubredditAvatarProps) {
  const image_src = "/subreddit_pics/" + subredditName + ".png";
  return (
    <img
      src={image_src}
      className="rounded-full w-12 h-12 md:w-15 md:h-15 object-cover"
    />
  );
}

export function DefaultSubredditAvatar() {
  const color = "#4268e8";

  return (
    <div
      className="flex items-center justify-center rounded-full w-12 h-12 md:w-15 md:h-15 font-semibold text-[30px]"
      style={{ backgroundColor: color }}
    >
      r/
    </div>
  );
}
