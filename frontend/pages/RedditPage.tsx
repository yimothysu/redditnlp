import { useState, useEffect, useRef } from "react";
import { useParams } from "react-router-dom";
import { ChevronLeft, ChevronRight } from "lucide-react";
import {
    fetchSubredditAnalysis,
    SubredditAnalysis,
    SubredditAnalysisResponse,
} from "../src/lib/api";
import { SubredditAvatar } from "../src/components/SubredditAvatar.tsx";

import Template from "./Template.tsx";
import WordEmbeddingsGraph from "../Components/WordEmbeddingsGraph.tsx";

interface AnalysisProgress {
    status: string;
    progress: number | null;
}

/*
const spinnerStyle = {
    border: "4px solid #f3f3f3",
    borderTop: "4px solid #3498db",
    borderRadius: "50%",
    width: "20px",
    height: "20px",
    animation: "spin 1s linear infinite",
};
*/

export default function RedditPage() {
    const { name } = useParams();
    const [error, setError] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [analysisProgress, setAnalysisProgress] =
        useState<AnalysisProgress | null>(null);
    const [analysis, setAnalysis] = useState<SubredditAnalysis | null>(null);
    const [timeFilter, setTimeFilter] = useState("week");
    const [sortBy, setSortBy] = useState("top");
    const [namedEntitiesExpandAllIsChecked, setNamedEntitiesExpandAllIsChecked] = useState(false)
    const [currNamedEntityCarouselIndex, setCurrNamedEntityCarouselIndex] = useState(0);
    const handleNamedEntitiesToggleChange = () => {
        setNamedEntitiesExpandAllIsChecked(!namedEntitiesExpandAllIsChecked)
    }

    const isMounted = useRef(true);

    useEffect(() => {
        isMounted.current = true;
        return () => {
            isMounted.current = false;
        };
    }, []);

    const loadSubredditAnalysis = async () => {
        if (isLoading) return;
        setIsLoading(true);
        try {
            while (isMounted.current) {
                const response: SubredditAnalysisResponse =
                    await fetchSubredditAnalysis({
                        name: name || "",
                        time_filter: timeFilter,
                        sort_by: sortBy,
                    });

                if (!isMounted.current) break;
                if (response.analysis) {
                    setAnalysisProgress(null);
                    setAnalysis(response.analysis);
                    break;
                }
                setAnalysisProgress({
                    status: response.analysis_status,
                    progress: response.analysis_progress,
                });

                if (!response.analysis_status) {
                    break;
                }
                await new Promise((resolve) => setTimeout(resolve, 1000));
            }
        } catch (err) {
            if (isMounted.current) {
                setError(
                    err instanceof Error
                        ? err.message
                        : "An unknown error occurred"
                );
                setAnalysisProgress(null);
            }
        } finally {
            if (isMounted.current) setIsLoading(false);
        }
    };

    const renderGenerationTimeEstimatesTable = () => {
        return (
            <div className="overflow-x-auto mt-7 mb-0 flex items-center justify-center">
                <table className="table-auto border-collapse text-sm m-2.5">
                    <thead>
                        <tr>
                            <th className="px-6 py-2 text-center text-left text-indigo-600">
                                Time Filter
                            </th>
                            <th className="px-8 py-2 text-center text-left text-indigo-600">
                                Approx Generation Time
                            </th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td className="px-4 py-1.5 text-center bg-gray-200">
                                Week
                            </td>
                            <td className="px-4 py-1.5 text-center bg-gray-200">
                                3-5 min
                            </td>
                        </tr>
                        <tr>
                            <td className="px-4 py-1.5 text-center">Month</td>
                            <td className="px-4 py-1.5 text-center">5-7 min</td>
                        </tr>
                        <tr>
                            <td className="px-4 py-1.5 text-center bg-gray-200">
                                Year
                            </td>
                            <td className="px-4 py-1.5 text-center bg-gray-200">
                                10 min
                            </td>
                        </tr>
                        <tr>
                            <td className="px-4 py-1.5 text-center">
                                All Time
                            </td>
                            <td className="px-4 py-1.5 text-center">15 min</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        );
    };

    const renderWordEmbeddings = () => {
        if (!analysis) return null;

        console.log('top_named_entities_embeddings: ', analysis?.top_named_entities_embeddings);
        let embeddings = analysis.top_named_entities_embeddings;
        let embeddingsDisplay = embeddings && Object.entries(embeddings).map(([word, [x, y]]) => ({ word, x, y,}));
        
        return (
            <div className="mt-10 p-6 bg-white shadow rounded max-w-3xl mx-auto">
                <h1 className="font-semibold text-l p-1">Word Embeddings</h1>
                <p className="text-gray-500 p-3">
                    Word embeddings are vector representations of words in
                    high-dimensional space, offering insights into meaning and
                    context. Below is a 2D projection of the most popular words
                    in the subreddit (reduced via PCA).
                </p>
                <div className="bg-gray-100 rounded">
                    {embeddingsDisplay && embeddingsDisplay.length > 0 ? (
                        <WordEmbeddingsGraph embeddings={embeddingsDisplay} />
                    ) : (
                        <p className="text-gray-400 p-4">No Embeddings</p>
                    )}
                </div>
            </div>
        );
    };

    const renderNGrams = () => {
        if (!analysis) return null;

        return (
            <div className="mt-4">
                <h2 className="text-xl font-bold mb-2">Top N-Grams</h2>
                <div className="grid sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-2">
                    {Object.entries(analysis.top_n_grams).map(
                        ([date, ngrams]) => (
                            <div
                                key={date}
                                className="mb-3 border bg-white border-gray-200 p-4 rounded-l shadow-md"
                            >
                                <h3 className="text-lg font-semibold">
                                    {date}
                                </h3>
                                <ul className="list-disc pl-5">
                                    {ngrams.map((ngram, index) => (
                                        <li key={index}>
                                            {ngram[0]}: {ngram[1]}
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        )
                    )}
                </div>
            </div>
        );
    };

    const renderNamedEntitiesExpandAllToggle = () => {
        return (
            <label className='autoSaverSwitch relative inline-flex cursor-pointer select-none items-center'>
                    <input type='checkbox' name='autoSaver' className='sr-only' checked={namedEntitiesExpandAllIsChecked} onChange={handleNamedEntitiesToggleChange}/>
                    <span className={`slider mr-3 flex h-[26px] w-[50px] items-center rounded-full p-1 duration-200 ${namedEntitiesExpandAllIsChecked ? 'bg-[#5a33ff]' : 'bg-[#CCCCCE]'}`}>
                        <span className={`dot h-[18px] w-[18px] rounded-full bg-white duration-200 ${namedEntitiesExpandAllIsChecked ? 'translate-x-6' : ''}`}></span>
                    </span>
                    <span className='label flex items-center text-sm font-medium text-black'> Expand All </span>
            </label>
        );
    };

    const renderNamedEntities = () => {
        if (!analysis) return null;

        const ColorCodeBox = (props) => {
            return (
                <div className="w-[150px] h-[35px] font-medium"
                     style={{fontSize: "15px", backgroundColor: props.backgroundColor, textAlign: "center",
                             padding: "5px", borderTopLeftRadius: props.borderTopLeftRadius, 
                             borderBottomLeftRadius: props.borderBottomLeftRadius, borderTopRightRadius: props.borderTopRightRadius,
                             borderBottomRightRadius: props.borderBottomRightRadius}}>
                    {props.label}
                </div>
            );
        };

        const TopNamedEntitiesForDate = (props) => {
            return (
            <div key={props.date} className="flex-1 mb-6 mr-2 ml-2 border bg-white border-gray-300 rounded-l shadow-xl">
                <h3 className="text-lg font-semibold" style={{fontSize: "20px", textAlign: "center", backgroundColor: "#efefef", padding: "4px",}}>
                    {props.date}
                </h3>
                <ul className="list-decimal pl-5">
                    {props.entities.map((entity, index) => {
                        const backgroundColor = entity[2] >= 0.5 && entity[2] <= 1 ? "#d9ead3"
                                                : entity[2] >= 0.1 && entity[2] < 0.5 ? "#d0e0e3"
                                                : entity[2] >= -0.1 && entity[2] < 0.1 ? "#fff2cc"
                                                : entity[2] >= -0.5 && entity[2] < -0.1 ? "#f4cccc"
                                                : entity[2] >= -1 && entity[2] < -0.5 ? "#ea9999"
                                                : "bg-gray-100";
                        return (
                            <li key={index} style={{ margin: "20px" }}>
                                <div style={{display: "flex", justifyContent:"space-between", alignItems: "center",
                                        backgroundColor: backgroundColor, paddingLeft: "10px", paddingRight: "10px",
                                        paddingTop: "4px", paddingBottom: "4px", borderRadius: "5px",}}>
                                    <span style={{fontWeight: 600, fontSize: "15px",}}>{entity[0]}</span>
                                    <span style={{margin: "0 8px",}}></span>
                                    <span style={{fontWeight: 600, fontSize: "15px",}}>Sentiment Score:{" "}{entity[2]}</span>
                                    <span style={{margin: "0 8px",}}></span>
                                    <span style={{fontWeight: 600, fontSize: "15px",}}># of Mentions:{" "}{entity[1]}</span>
                                </div>
                                <div style={{fontWeight: 600, fontSize: "13.5px", marginTop: "10px", marginBottom: "5px",}}>
                                    Summarized Sentiment:{" "}
                                </div>
                                <div style={{fontSize: "13.5px", color: "#333",}}>{entity[3]}</div>
                            </li>);})}
                </ul>
            </div>
            );
        };

        const incrementCurrNamedEntityCarouselIndex = () => {
            setCurrNamedEntityCarouselIndex(currNamedEntityCarouselIndex + 1)
        };

        const decrementCurrNamedEntityCarouselIndex = () => {
            setCurrNamedEntityCarouselIndex(currNamedEntityCarouselIndex - 1)
        };

        const renderLeftCarouselButton = () => {
            return (
                <div style={{ position: "absolute", left: "0" }}>{currNamedEntityCarouselIndex != 0 && 
                    <button onClick={decrementCurrNamedEntityCarouselIndex} className="p-1 rounded-full text-white mr-30 bg-black transition hover:bg-gray-600">
                        <ChevronLeft size={32}/>
                    </button>}
                </div>
            );
        };

        const renderRightCarouselButton = () => {
            const topNamedEntitiesLength = Object.keys(analysis.top_named_entities).length;
            return (
                <div style={{ position: "absolute", right: "0"  }}>{currNamedEntityCarouselIndex < topNamedEntitiesLength - 2 && 
                    <button onClick={incrementCurrNamedEntityCarouselIndex} className="p-1 rounded-full text-white ml-30 bg-black transition hover:bg-gray-600">
                        <ChevronRight size={32}/>
                    </button>}
                </div>
            );
        };

        return (
            <div className="mt-7">
                <div className="flex justify-between items-center w-full relative">
                    <h2 className="text-xl font-medium mb-2 text-center absolute left-1/2 transform -translate-x-1/2" style={{ textAlign: "center", fontSize: "20px" }}>
                    Most Mentioned Named Entities
                    </h2>
                    <div className="ml-auto mr-8">{renderNamedEntitiesExpandAllToggle()}</div>
                </div>
                <div style={{margin: "25px", marginBottom: "40px", display: "flex", alignItems: "center", justifyContent: "center", position: "relative" }}>
                    {renderLeftCarouselButton()}
                    <div className="flex justify-center">
                        <ColorCodeBox backgroundColor="#d9ead3" label="Very Positive" borderTopLeftRadius="5px" borderBottomLeftRadius="5px"></ColorCodeBox>
                        <ColorCodeBox backgroundColor="#d0e0e3" label="Positive"></ColorCodeBox>
                        <ColorCodeBox backgroundColor="#fff2cc" label="Neutral"></ColorCodeBox>
                        <ColorCodeBox backgroundColor="#f4cccc" label="Negative"></ColorCodeBox>
                        <ColorCodeBox backgroundColor="#ea9999" label="Very Negative" borderBottomRightRadius="5px" borderTopRightRadius="5px"></ColorCodeBox>
                    </div>
                    {renderRightCarouselButton()}
                </div>
                <div className="overflow-hidden w-full">
                    <div className="flex transition-transform duration-600 ease-in-out"
                        style={{ transform: `translateX(-${currNamedEntityCarouselIndex * 50}%)` }}>
                        {Object.entries(analysis.top_named_entities).map(([date, entities]) => (
                            <div key={date} className="w-1/2 flex-shrink-0">
                                <TopNamedEntitiesForDate date={date} entities={entities} />
                            </div>
                        ))}
                     </div>
                </div>
            </div>
        );
    };

    // <div className="spinner" style={spinnerStyle}></div>

    return (
        <Template>
            <div className="m-4 p-4 bg-white rounded-md">
                <div className="flex flex-col items-center mb-6">
                    <SubredditAvatar subredditName={name ?? ""} />
                    <h1 className="text-lg font-bold">r/{name}</h1>
                </div>
                <div className="flex gap-25 justify-center">
                    <div className="mb-4 flex items-center justify-center">
                        <div className="flex gap-4 mb-2">
                            <div>
                                <label className="block text-sm font-medium text-gray-700">
                                    Time Filter
                                </label>
                                <select
                                    value={timeFilter}
                                    onChange={(e) =>
                                        setTimeFilter(e.target.value)
                                    }
                                    className="mt-1 block w-full px-1 py-2 text-base border border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                                >
                                    <option value="all">All Time</option>
                                    <option value="year">Year</option>
                                    <option value="month">Month</option>
                                    <option value="week">Week</option>
                                </select>
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-gray-700">
                                    Sort By
                                </label>
                                <select
                                    value={sortBy}
                                    onChange={(e) => setSortBy(e.target.value)}
                                    className="mt-1 block w-full px-1 py-2 text-base border border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                                >
                                    <option value="top">Top</option>
                                    <option value="controversial">
                                        Controversial
                                    </option>
                                </select>
                            </div>
                            <button
                                onClick={() => {
                                    loadSubredditAnalysis();
                                }}
                                className="block self-end px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 hover:cursor-pointer focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                            >
                                Analyze
                            </button>
                        </div>
                    </div>
                    {renderGenerationTimeEstimatesTable()}
                </div>

                {analysisProgress && (
                    <div className="flex flex-col items-center py-8">
                        <div className="flex items-center justify-center space-x-2">
                            <h2>{analysisProgress.status}</h2>
                        </div>
                        {analysisProgress.progress && (
                            <div className="w-40 h-2 mt-4 flex items-center justify-center">
                                <div className="relative w-full h-full bg-gray-200 rounded-lg overflow-hidden">
                                    <div
                                        className="h-full bg-blue-500 transition-all duration-500 ease-out"
                                        style={{
                                            width: `${
                                                100 * analysisProgress.progress
                                            }%`,
                                        }}
                                    />
                                </div>
                            </div>
                        )}
                    </div>
                )}

                {error && (
                    <div className="bg-red-50 border-l-4 border-red-400 p-4 mt-4">
                        <div className="flex">
                            <div className="ml-3">
                                <p className="text-sm text-red-700">
                                    Error: {error}
                                </p>
                            </div>
                        </div>
                    </div>
                )}

                {!analysisProgress && !error && analysis && (
                    <div className="mt-4">
                        <hr className="my-4 border-t border-gray-300 mx-auto w-[97%]" />
                        {renderNamedEntities()}
                        {renderNGrams()}
                        {renderWordEmbeddings()}
                    </div>
                )}
            </div>
        </Template>
    );
}
