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
import { render } from "react-dom";
import Button from '@mui/material/Button';
import Popover from '@mui/material/Popover';
import { Typography } from "@mui/material";
import PopupState, { bindTrigger, bindPopover } from 'material-ui-popup-state';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";


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

const months = {
    '01': 'January','02': 'February','03': 'March','04': 'April','05': 'May','06': 'June','07': 'July','08': 'August',
    '09': 'September','10': 'October','11': 'November','12': 'December'
};

function formatDate(date, time_filter) {
    const currentYear = new Date().getFullYear();
    if (time_filter == 'week') {
        // Ex: 03-13 --> May 13, 2025
        return months[date.slice(0, 2)] + ' ' + date.slice(3, 5) + ', ' + currentYear
    }
    else if (time_filter == 'year') {
        // Ex: 03-24 --> May 2024 
        return months[date.slice(0, 2)] + ' 20' + date.slice(3, 5)
    }
    return date 
}

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
    const [currNGramsCarouselIndex, setCurrNGramsCarouselIndex] = useState(0);
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

    function createHistogram(values) {
        console.log('inside createHistogram')
        console.log(values)
        const minVal = 0;
        const maxVal = Math.max(...values);
        const num_bins = 30
        const binWidth = (maxVal - minVal) / num_bins;
    
        const histogram = Array.from({ length: num_bins }, () => ({ bin: 0, count: 0 }));
  
        values.forEach(value => {
            const bin_index = Math.min(Math.floor((value - minVal) / binWidth), num_bins - 1); 
            histogram[bin_index].count += 1;
        });
  
        return histogram.map((d, i) => ({
            bin: (minVal + i * binWidth).toFixed(2),
            count: d.count
        }));
    };

    const renderComparativeAnalysis = () => {
        if (!analysis) return null;

        const ToxicityPopoverButton = () => {
            return (
                <PopupState variant="popover" popupId="demo-popup-popover">
                  {(popupState) => (
                    <div>
                      <Button sx={{ backgroundColor: "#719efd", mt: 2, mb: 3, fontWeight: "bold", fontSize: "15px"}}  variant="contained" {...bindTrigger(popupState)}>
                        What is Toxicity Score?
                      </Button>
                      <Popover
                        {...bindPopover(popupState)}
                        anchorOrigin={{
                          vertical: 'bottom',
                          horizontal: 'center',
                        }}
                        transformOrigin={{
                          vertical: 'top',
                          horizontal: 'center',
                        }}
                      >
                        <Typography sx={{ p: 2, maxWidth: 500, fontWeight: 'bold', fontSize: '18px', textAlign: 'center', color: '#FF0000'}}> Toxicity Score is a float in the range [0, 1]. 
                       </Typography>
                       <Typography sx={{ p: 1, maxWidth: 500, fontWeight: 'bold', fontSize: '18px', textAlign: 'center', color: '#FF0000'}}>0 = no toxic content, 1 = all toxic content</Typography>
                       <hr className="border-t-3 border-gray-300 my-4 ml-5 mr-5" ></hr>
                        <Typography sx={{ p: 1, maxWidth: 500, textAlign: 'center', fontWeight: 'bold'}}>How is Toxicity Score calculated?</Typography>
                        <Typography sx={{ p: 1, maxWidth: 500}}>400 top posts of all time from r/{name} are sampled. Only the TITLE 
                            and DESCRIPTION of the posts are analyzed. 
                       </Typography>
                       <Typography sx={{ p: 1, maxWidth: 500}}>The python library 'Detoxify' analyzes all 
                            the post text and scores the text on 6 metrics: toxicity, severe_toxicity, obscene, threat, insult,
                            identify attack. A weighted average of these 6 metrics is taken to get an overall toxicity score.
                       </Typography>
                       <Typography sx={{ p: 1, maxWidth: 500, fontWeight: 'bold'}}>
                            NOTE: 'Detoxify' analyzes text LITERALLY. It cannot grasp humor or sarcasm (so dark jokes will have a high
                            toxic score)
                       </Typography>
                      </Popover>
                    </div>
                  )}
                </PopupState>
              );
          }

          const PositiveContentPopoverButton = () => {
            return (
                <PopupState variant="popover" popupId="demo-popup-popover">
                  {(popupState) => (
                    <div>
                      <Button sx={{ backgroundColor: "#719efd", mt: 2, mb: 3, fontWeight: "bold", fontSize: "15px"}}  variant="contained" {...bindTrigger(popupState)}>
                        What is Positive Content Score?
                      </Button>
                      <Popover
                        {...bindPopover(popupState)}
                        anchorOrigin={{
                          vertical: 'bottom',
                          horizontal: 'center',
                        }}
                        transformOrigin={{
                          vertical: 'top',
                          horizontal: 'center',
                        }}
                      >
                        <Typography sx={{ p: 2, maxWidth: 500, fontWeight: 'bold', fontSize: '18px', textAlign: 'center', color: '#FF0000'}}> Positive Content Score is a float in the range [0, 1]. 
                       </Typography>
                       <Typography sx={{ p: 1, maxWidth: 500, fontWeight: 'bold', fontSize: '18px', textAlign: 'center', color: '#FF0000'}}>0 = no positive content, 1 = all positive content</Typography>
                       <hr className="border-t-3 border-gray-300 my-4 ml-5 mr-5" ></hr>
                        <Typography sx={{ p: 1, maxWidth: 500, textAlign: 'center', fontWeight: 'bold'}}>How is Positive Content Score calculated?</Typography>
                        <Typography sx={{ p: 2, maxWidth: 500}}>400 top posts of all time from r/{name} are sampled. Only the TITLE 
                            and DESCRIPTION of the posts are analyzed. 
                       </Typography>
                       <Typography sx={{ p: 2, maxWidth: 500}}>The python class 'nltk.sentiment.vader.SentimentIntensityAnalyzer' 
                            analyzes all the post text and scores the text from -1 to 1 (with -1 being most negative, 0 being 
                            neutral, and 1 being most positive). This score is normalized to [0, 1]. 
                       </Typography>
                       <Typography sx={{ p: 2, maxWidth: 500}}>
                            Posts that have more upvotes have a greater effect on the overall positive content score (i.e. the positive
                            content score is a weighted average)
                       </Typography>
                      </Popover>
                    </div>
                  )}
                </PopupState>
              );
          }
        
        const GradeCircle = ({grade}) => {
            const getGradeBackgroundColor = () => {
                if (['A+', 'A', 'A-'].includes(grade)) { return 'bg-green-300'; }
                if (['B+', 'B', 'B-'].includes(grade)) { return 'bg-lime-300'; }
                if (['C+', 'C', 'C-'].includes(grade)) { return 'bg-yellow-300'; }
                if (['D+', 'D', 'D-'].includes(grade)) { return 'bg-orange-300'; }
                if (['F'].includes(grade)) { return 'bg-red-300'; }
            };
        

            const getGradeBorderColor = () => {
                if (['A+', 'A', 'A-'].includes(grade)) { return 'border-4 border-green-400'; }
                if (['B+', 'B', 'B-'].includes(grade)) { return 'border-4 border-lime-400'; }
                if (['C+', 'C', 'C-'].includes(grade)) { return 'border-4 border-yellow-400'; }
                if (['D+', 'D', 'D-'].includes(grade)) { return 'border-4 border-orange-400'; }
                if (['F'].includes(grade)) { return 'border-4 border-red-500'; }
            };

            return (
                <span className={`inline-flex items-center justify-center w-13 h-13 ${getGradeBackgroundColor()} 
                    ${getGradeBorderColor()} text-black text-[20px] m-3 rounded-full`}>{grade}</span>
            );
        }

        const Histogram = ({values, xaxis_label}) => {
            //console.log(values)
            const histogramData = createHistogram(values);

            return (
              <div className="mt-5 h-90 w-120 p-4 bg-white shadow-md rounded-xl">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={histogramData} margin={{ bottom: 15, left: 7}}>
                    <XAxis dataKey="bin" tick={{ fontSize: 10, fill: 'black' }} label={{ value: xaxis_label, fill: 'black', position: "outsideBottom", dy: 20 }}/>
                    <YAxis tick={{ fill: "black" }} label={{ value: "Count", fill: 'black', angle: -90, position: "insideLeft", dy: 20, dx: -4 }}/>
                    <Tooltip />
                    <Bar dataKey="count" fill="#4f46e5"></Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            );
        };

        return (
            <div style={{margin: "15px", marginBottom: "30px"}}>
                {/* <h2 className="text-xl text-center font-bold">How does r/{name} compare to other subreddits?</h2> */}
                <div className="flex justify-between ml-10 mt-7 mr-10">
                    <div className="items-center">
                        <div className="text-center">
                            <h1 className="text-xl font-bold">Ranking r/{name}'s Toxicity Score</h1>
                            <ToxicityPopoverButton></ToxicityPopoverButton>
                        </div>
                        <ul className="list-disc pl-5 ml-25">
                            <li><h1 className="text-[16px] font-semibold m-1">r/{name}'s toxicity score = {Number(analysis.toxicity_score.toFixed(2))}</h1></li>
                            <li><h1 className="text-[16px] font-semibold m-1">r/{name}'s toxicity grade = 
                                    <GradeCircle grade={analysis.toxicity_grade}></GradeCircle>
                                </h1>
                            </li>
                            <li>
                                <h1 className="text-[16px] font-semibold m-1">r/{name}'s toxicity percentile = {Number(analysis.toxicity_percentile.toFixed(1))}%</h1>
                                <h1 className="text-[16px] max-w-xs m-1 italic">This means that r/{name} has a higher toxicity score than {Number(analysis.toxicity_percentile.toFixed(1))}% 
                                    of the 4000+ sampled subreddits </h1>
                            </li>
                        </ul>
                        <Histogram values={analysis.all_toxicity_scores} xaxis_label="Toxicity Score"></Histogram>
                        {/* <h1 className="mt-3 text-[15px] text-center text-red-500">Red bar is where r/{name} lies in the distribution</h1> */}
                    </div>
                    <div className="h-auto w-[2px] bg-gray-200"></div>
                    <div className="items-center">
                        <div className="text-center">
                            <h1 className="text-xl font-bold">Ranking r/{name}'s Positive Content Score</h1>
                            <PositiveContentPopoverButton></PositiveContentPopoverButton>
                        </div>
                        <ul className="list-disc pl-5 ml-25">
                            <li><h1 className="text-[16px] font-semibold m-1">r/{name}'s positive content score = {Number(analysis.positive_content_score.toFixed(2))}</h1></li>
                            <li><h1 className="text-[16px] font-semibold m-1">r/{name}'s positive content grade = 
                                    <GradeCircle grade={analysis.positive_content_grade}></GradeCircle>
                                </h1>
                            </li>
                            <li>
                                <h1 className="text-[16px] font-semibold m-1">r/{name}'s positive content percentile = {Number(analysis.positive_content_percentile.toFixed(1))}%</h1>
                                <h1 className="text-[16px] max-w-xs m-1 italic">This means that r/{name} has a higher positive content score than {Number(analysis.positive_content_percentile.toFixed(1))}% 
                                    of the 4000+ sampled subreddits</h1>
                            </li>
                        </ul>
                        <Histogram values={analysis.all_positive_content_scores} xaxis_label="Positive Content Score"></Histogram>
                        {/* <h1 className="mt-3 text-[15px] text-center text-red-500">Red bar is where r/{name} lies in the distribution</h1> */}
                    </div>
                </div>
               
            </div>
        )
    }

    const renderNGrams = () => {
        if (!analysis) return null;

        const NGramsForDate = ({date, ngrams}) => {
            return (
            <div key={date} className="flex-grow mb-3 mr-1 ml-1 border bg-white border-gray-200 rounded-l shadow-xl">
                <h3 className="text-lg text-center p-1 font-semibold bg-gray-200">{formatDate(date, timeFilter)}</h3>
                <div className="flex font-semibold pt-1 pb-1 pr-5 pl-5 bg-blue-200 justify-between">
                    <h2>Word</h2>
                    <h2>Count</h2>
                </div>
                <ul className="list-none pl-5 pr-5 pt-2">
                    {ngrams.map((ngram, index) => (
                        <li key={index}>
                            <div className='text-[14px] flex justify-between'>
                                <div>{ngram[0]}</div> 
                                <div>{ngram[1]}</div>
                            </div>
                            <hr className="border-t border-gray-300 my-1" />
                        </li>))}
                </ul>
            </div>
            );
        };

        const incrementCurrNGramsCarouselIndex = () => {
            setCurrNGramsCarouselIndex(currNGramsCarouselIndex + 1)
        };

        const decrementCurrNGramsCarouselIndex = () => {
            setCurrNGramsCarouselIndex(currNGramsCarouselIndex - 1)
        };

        const renderLeftCarouselButton = () => {
            return (
                <div style={{ position: "absolute", left: "0" }}>{currNGramsCarouselIndex != 0 && 
                    <button onClick={decrementCurrNGramsCarouselIndex} className="p-1 rounded-full text-white mr-30 bg-black transition hover:bg-gray-600">
                        <ChevronLeft size={28}/>
                    </button>}
                </div>
            );
        };

        const renderRightCarouselButton = () => {
            const topNGramsLength = Object.keys(analysis.top_n_grams).length;
            return (
                <div style={{ position: "absolute", right: "0"  }}>{currNGramsCarouselIndex < topNGramsLength - 5 && 
                    <button onClick={incrementCurrNGramsCarouselIndex} className="p-1 rounded-full text-white ml-30 bg-black transition hover:bg-gray-600">
                        <ChevronRight size={28}/>
                    </button>}
                </div>
            );
        };

        return (
            <div className="mt-4">
                <div style={{margin: "25px", marginBottom: "30px", display: "flex", alignItems: "center", justifyContent: "center", position: "relative" }}>
                    {renderLeftCarouselButton()}
                    <h2 className="text-xl text-center font-bold">Most Mentioned Bi-Grams</h2>
                    {renderRightCarouselButton()}
                </div>
                <div className="overflow-hidden w-full">
                    <div className="flex transition-transform duration-600 ease-in-out"
                        style={{ transform: `translateX(-${currNGramsCarouselIndex * 20}%)` }}>
                        {Object.entries(analysis.top_n_grams).map(([date, ngrams]) => (
                            <div key={date} className="flex w-1/5 flex-shrink-0">
                                <NGramsForDate date={date} ngrams={ngrams } />
                            </div>
                        ))}
                     </div>
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
                <div className="w-[180px] h-[35px] font-medium"
                     style={{fontSize: "14px", backgroundColor: props.backgroundColor, textAlign: "center", 
                             padding: "5px", borderTopLeftRadius: props.borderTopLeftRadius, 
                             borderBottomLeftRadius: props.borderBottomLeftRadius, borderTopRightRadius: props.borderTopRightRadius,
                             borderBottomRightRadius: props.borderBottomRightRadius}}>
                    {props.label}
                </div>
            );
        };

        const TopNamedEntitiesForDate = (props) => {
            return (
            <div key={props.date} className="flex-grow flex-1 mb-6 mr-1 ml-1 border bg-white border-gray-300 rounded-l shadow-xl">
                <h3 className="text-lg font-semibold bg-gray-200" style={{fontSize: "20px", textAlign: "center", padding: "4px",}}>
                    {formatDate(props.date, timeFilter)}
                </h3>
                <ul className="list-none pl-0">
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
                                    <span style={{fontWeight: 600, fontSize: "14.5px",}}>{entity[0]}</span>
                                    <span style={{margin: "0 8px",}}></span>
                                    <span style={{fontWeight: 600, fontSize: "13.5px",}}>Sentiment Score [-1, 1]:{" "}{entity[2]}</span>
                                    <span style={{margin: "0 8px",}}></span>
                                    <span style={{fontWeight: 600, fontSize: "13.5px",}}># Mentions:{" "}{entity[1]}</span>
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
                <div style={{ position: "absolute", right: "0"  }}>{currNamedEntityCarouselIndex < topNamedEntitiesLength - 3 && 
                    <button onClick={incrementCurrNamedEntityCarouselIndex} className="p-1 rounded-full text-white ml-30 bg-black transition hover:bg-gray-600">
                        <ChevronRight size={32}/>
                    </button>}
                </div>
            );
        };

        return (
            <div className="mt-7">
                <div className="flex justify-between items-center w-full relative">
                    <h2 className="text-xl font-bold mt-3 mb-2 text-center absolute left-1/2 transform -translate-x-1/2" style={{ textAlign: "center", fontSize: "20px" }}>
                    Most Mentioned Named Entities
                    </h2>
                    <div className="ml-auto mr-8">{renderNamedEntitiesExpandAllToggle()}</div>
                </div>
                {/* <div className="flex flex-col mt-3 items-center space-y-4">
                    <Button variant="outlined" sx={{ width: '500px', borderColor: "#719efd", mt: 2, fontWeight: "bold", fontSize: "14px"}}>How is a sentiment score computed for each entity?</Button>
                    <Button variant="outlined" sx={{ width: '500px', borderColor: "#719efd", mt: 2, fontWeight: "bold", fontSize: "14px"}}>How is summarized sentiment computed for each entity?</Button>
                </div> */}
                <div style={{margin: "25px", marginBottom: "40px", display: "flex", alignItems: "center", justifyContent: "center", position: "relative" }}>
                    {renderLeftCarouselButton()}
                    <div className="flex justify-center">
                        <ColorCodeBox backgroundColor="#d9ead3" label="Very Positive Consensus" borderTopLeftRadius="5px" borderBottomLeftRadius="5px"></ColorCodeBox>
                        <ColorCodeBox backgroundColor="#d0e0e3" label="Positive Consensus"></ColorCodeBox>
                        <ColorCodeBox backgroundColor="#fff2cc" label="Neutral Consensus"></ColorCodeBox>
                        <ColorCodeBox backgroundColor="#f4cccc" label="Negative Consensus"></ColorCodeBox>
                        <ColorCodeBox backgroundColor="#ea9999" label="Very Negative Consensus" borderBottomRightRadius="5px" borderTopRightRadius="5px"></ColorCodeBox>
                    </div>
                    {renderRightCarouselButton()}
                </div>
                <div className="overflow-hidden w-full">
                    <div className="flex transition-transform duration-600 ease-in-out"
                        style={{ transform: `translateX(-${currNamedEntityCarouselIndex * 33}%)` }}>
                        {Object.entries(analysis.top_named_entities).map(([date, entities]) => (
                            <div key={date} className="flex w-1/3 flex-shrink-0">
                                <TopNamedEntitiesForDate date={date} entities={entities} />
                            </div>
                        ))}
                     </div>
                </div>
                <div className="flex flex-col mt-3 items-center space-y-4">
                    <Button variant="outlined" sx={{ width: '500px', borderColor: "#719efd", mt: 2, fontWeight: "bold", fontSize: "14px"}}>How is a sentiment score computed for each entity?</Button>
                    <Button variant="outlined" sx={{ width: '500px', borderColor: "#719efd", mt: 2, fontWeight: "bold", fontSize: "14px"}}>How is summarized sentiment computed for each entity?</Button>
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
                        {renderComparativeAnalysis()}
                        <hr className="my-4 border-t border-gray-300 mx-auto w-[97%]" />
                        {renderNGrams()}
                        <hr className="my-4 border-t border-gray-300 mx-auto w-[97%]" />
                        {renderNamedEntities()}
                        {renderWordEmbeddings()}
                    </div>
                )}
            </div>
        </Template>
    );
}
