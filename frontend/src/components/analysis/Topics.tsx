import { SubredditAnalysis } from "../../types/redditAnalysis";
import { formatDistanceToNow } from 'date-fns';
import { useState } from "react";

interface TopicsProps {
    analysis: SubredditAnalysis;
    timeFilter: string;
  }

export const Topics: React.FC<TopicsProps> = ({ analysis }) => {
  const [selectedTopic, setSelectedTopic] = useState(Object.keys(analysis.topics)[0]);

  return (
    <div className="flex flex-row">
        <div className="w-60 flex-shrink-0 rounded-lg">
            {Object.entries(analysis.topics).map(([topic, _]) => (
                <div>
                    <div onClick={() => setSelectedTopic(topic)} className={`pt-3 pb-1 text-center text-gray-400 group text-xs rounded font-semibold ${topic !== selectedTopic ? 'hover:bg-gray-100' : ''} ${topic === selectedTopic ? 'bg-[#fa6f4d] text-white' : ''}`}>
                        <div className={`${topic !== selectedTopic ? 'group-hover:text-gray-600' : ''}`}>
                            {topic.toLowerCase()}
                        </div>
                    </div>
                    <hr className="border-gray-200"></hr>
                </div>
            ))}
        </div>
        <div className="ml-4">
            <div key={selectedTopic} className="pb-4">
            <hr className="border-t border-gray-300 opacity-50"></hr>
            <h2 className="p-1 mb-2 font-semibold font-[14px] bg-gray-100 text-gray-500 tracking-wide text-center">{selectedTopic.toLowerCase()}</h2>
            <ul className="list-disc list-inside ml-4">
                {analysis.topics[selectedTopic].map((post, _) => (
                    <div className="text-sm pb-7">
                        <div className="flex flex-row">
                            <div className="font-normal text-xs text-gray-600 mt-2 mr-3">
                                {formatDistanceToNow(new Date(post.created_utc * 1000), { addSuffix: true })} 
                            </div>
                            <div className="flex flex-row bg-gray-200 text-xs font-semibold rounded-full p-1 text-center">
                                <svg width="22" height="22" viewBox="0 0 24 24" strokeLinecap="round" strokeLinejoin="round" className="fill-gray-200 outline:black stroke-black stroke-1 cursor-pointer">
                                    <path d="M12 4l-8 8h5v8h6v-8h5z" />
                                </svg>
                                <div className="ml-1 mt-1 mr-2">{post.score}</div> 
                                <svg width="22" height="22" viewBox="0 0 24 24" strokeLinecap="round" strokeLinejoin="round" className="fill-gray-200 outline:black stroke-black stroke-1 cursor-pointer">
                                    <path d="M12 20l8-8h-5v-8h-6v8h-5z" />
                                </svg>
                            </div>
                            <div className="flex flex-row bg-gray-200 rounded-full p-1 ml-3">
                                <svg
                                    className="w-5 h-5 fill-gray-200 outline:black stroke-black ml-1 mt-1"
                                    viewBox="0 0 24 24"
                                    preserveAspectRatio="none"
                                    xmlns="http://www.w3.org/2000/svg"
                                >
                                <path d="M20 2H4C2 2 2 4 2 4v12c0 2 2 2 2 2h4v3.17c0 .53.64.8 1.01.42L12.6 18H20c2 0 2-2 2-2V4c0-2-2-2-2-2z" />
                                </svg>
                                <div className="ml-1 mt-1 mr-1 font-semibold text-xs">
                                    {post.num_comments} 
                                </div>
                            </div>
                            <div className="cursor-pointer hover:bg-[#f25a3a] flex bg-[#fa6f4d] active:bg-[#e76345] font-normal rounded-full text-white p-1 pt-2 ml-3 text-xs" onClick={() => window.open(post.url, "_blank")}>
                                read post
                                <img
                                    src={"/link_icon.png"}
                                    className="ml-1 rounded-full w-4 h-4 object-cover cursor-pointer transition active:scale-95 active:brightness-90"
                                />
                            </div>
                        </div>
                        <div className="mt-1 font-semibold">
                            {post.title}
                        </div>
                    </div>
                ))}
          </ul>
        </div>
    </div>
    </div>
  );
};
