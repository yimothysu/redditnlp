import { useState, useEffect, useRef } from "react";
import {
    NamedEntity,
    NGram,
    fetchSubredditAnalysis,
    SubredditAnalysis,
} from "../src/lib/api";
import { SubredditAvatar } from "../src/components/SubredditAvatar.tsx";

import WordEmbeddingsGraph from "./WordEmbeddingsGraph.tsx";
import { Typography } from "@mui/material";
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    Tooltip,
    ResponsiveContainer,
    Cell,
} from "recharts";
import ButtonPopover from "./ButtonPopover.tsx";

const months: Record<string, string> = {
    "01": "January",
    "02": "February",
    "03": "March",
    "04": "April",
    "05": "May",
    "06": "June",
    "07": "July",
    "08": "August",
    "09": "September",
    "10": "October",
    "11": "November",
    "12": "December",
};

function formatDate(date: string, time_filter: string) {
    //const currentYear = new Date().getFullYear();
    if (time_filter == "week") {
        // Ex: 03-13 --> May 13, 2025
        return (
            months[date.slice(0, 2)] +
            " " +
            date.slice(3, 5) 
            // + ", "  + currentYear
        );
    } else if (time_filter == "year") {
        // Ex: 03-24 --> May 2024
        return months[date.slice(0, 2)] + " 20" + date.slice(3, 5);
    }
    else if (time_filter == "all") {
        return "20" + date;
    }
    return date;
}

type Props = {
    name?: string;
    inComparisonMode: string; 
};

export default function RedditAnalysisDisplay({ name, inComparisonMode }: Props) {

    const [error, setError] = useState<string | null>(null);
    const [analysis, setAnalysis] = useState<SubredditAnalysis | null>(null);
    const [currentMenuItem, setCurrentMenuItem] = useState("Named Entities");
    const [currentNamedEntityDate, setCurrentNamedEntityDate] = useState("");
    const [isBusy, setIsBusy] = useState(false);
    const [timeFilter, setTimeFilter] = useState("week");
    const [sortBy, setSortBy] = useState("top");
    

    const isMounted = useRef(true);
    
    useEffect(() => {
        setAnalysis(null); // reset when timeFilter changes
        loadSubredditAnalysis();
    }, [timeFilter]);
    

    useEffect(() => {
        isMounted.current = true;
        return () => {
            isMounted.current = false;
        };
    }, []);
    

    const loadSubredditAnalysis = async () => {
        if (isBusy || !name || name.trim().length === 0) return;
        setIsBusy(true);
        try {
            const analysis: SubredditAnalysis = await fetchSubredditAnalysis({
                name: name.trim(),
                time_filter: timeFilter,
                sort_by: sortBy,
            });
            setError(null)
            setCurrentNamedEntityDate(Object.keys(analysis.top_named_entities)[0])
            setAnalysis(analysis);
        } catch (error) {
            setError(
                error instanceof Error
                    ? error.message
                    : "An unknown error occurred"
            );
        } finally {
            setIsBusy(false);
        }
    };

     // Call loadSubredditAnalysis only once when the component is mounted
     useEffect(() => {
        loadSubredditAnalysis();
    }, []); // Empty dependency array ensures it's called only once on mount

    const renderReadabilityMetrics = () => {

        if (!analysis) return null;
        console.log(analysis.readability_metrics)
        return (
            <div>
                <h1 className="font-bold text-xl text-center mt-5 p-1">Readability Metrics</h1>
                <br/>
                <div className="text-center">
                    <ButtonPopover title="What are readability metrics">
                        Readability metrics evaluate how easy a body text is to read. This includes expected grade level of education required to understand a text.
                    </ButtonPopover>
                </div>
                <br/>
                <p className="text-center text-gray-800"><b>Average Number of Words Per Post Title: </b>{analysis.readability_metrics["avg_num_words_title"]}</p>
                <p className="text-center text-gray-800"><b>Average Number of Words Per Post Description: </b>{analysis.readability_metrics["avg_num_words_description"]}</p>
                <p className="text-center text-gray-800"><b>Average Grade Level of Text (Flesch Score): </b>
                    {analysis.readability_metrics["avg_flesh_grade_level"] != -1 ? analysis.readability_metrics["avg_flesh_grade_level"].toFixed(2) : "Not enough data to provide"}
                </p>
                <p className="text-center"><b>Average Grade Level of Text (Dale Chall Score): </b>
                    {analysis.readability_metrics["avg_dale_chall_grade_level"] != -1 ? analysis.readability_metrics["avg_dale_chall_grade_level"].toFixed(2) : "Not enough data to provide"}
                </p>
            </div>);
    }

    const renderWordCloud = () => {
        if (!analysis) return null;

        const wordCloud = new Image();
        wordCloud.src = `data:image/png;base64, ${analysis.top_named_entities_wordcloud}`;

        return (
            <div className="flex flex-col items-center">
            <h1 className="font-bold text-xl text-center mt-4 mb-4 p-1">Word Cloud</h1>
            <ButtonPopover title="What's a Word Cloud?'">
                A word cloud is a visual representation of the most popular words in the subreddit. The size of the word corresponds to its frequency in the subreddit.
            </ButtonPopover>
            <div className="mt-4 p-2 pl-6 pr-6 pb-6 bg-white w-full mx-auto">
                {analysis?.top_named_entities_wordcloud ? (
                    <img
                        className="mx-auto w-[50%] h-auto"
                        src={wordCloud.src}
                        alt="Word Cloud"
                    />
                ) : (
                    <p className="text-gray-400 p-4">No Word Cloud</p>
                )}
            </div>
            </div>
        );
    };

    const renderWordEmbeddings = () => {
        if (!analysis) return null;

        console.log(
            "top_named_entities_embeddings: ",
            analysis?.top_named_entities_embeddings
        );
        let embeddings = analysis.top_named_entities_embeddings;
        let embeddingsDisplay =
            embeddings &&
            Object.entries(embeddings).map(([word, [x, y]]) => ({
                word,
                x: x * 5,
                y: y * 5,
            }));


        return (
            <div className="flex flex-col items-center" style={{ width: '100%', height: 'auto'}}>
            <h1 className="font-bold text-xl mt-4 text-center p-1 mb-4">Word Embeddings</h1>
            <ButtonPopover title="What are word embeddings?">
                Word embeddings are vector representations of words in
                high-dimensional space, offering insights into meaning and
                context. Below is a 2D projection of the most popular words
                in the subreddit (reduced via PCA).
            </ButtonPopover>
            <div className="mt-4 p-2 pl-6 pr-6 pb-6 bg-white">
                <div className="bg-gray-100 rounded">
                    {embeddingsDisplay && embeddingsDisplay.length > 0 ? (
                        <WordEmbeddingsGraph embeddings={embeddingsDisplay} isComparisonMode={inComparisonMode}/>
                    ) : (
                        <p className="text-gray-400 p-4">No Embeddings</p>
                    )}
                </div>
            </div>
            </div>
        );
    };

    function createHistogram(values: number[]) {
        console.log("inside createHistogram");
        console.log(values);
        const minVal = 0;
        const maxVal = Math.max(...values);
        const num_bins = 30;
        const binWidth = (maxVal - minVal) / num_bins;

        const histogram = Array.from({ length: num_bins }, () => ({
            bin: 0,
            count: 0,
        }));

        values.forEach((value) => {
            const bin_index = Math.min(
                Math.floor((value - minVal) / binWidth),
                num_bins - 1
            );
            histogram[bin_index].count += 1;
        });

        return histogram.map((d, i) => ({
            bin: (minVal + i * binWidth).toFixed(2),
            count: d.count,
        }));
    }

    const renderComparativeAnalysis = () => {
        if (!analysis) return null;

        const PositiveContentPopoverButton = () => {
            return (
                            <ButtonPopover title="What's Positive Content Score?">
                                <Typography
                                    sx={{
                                        p: 2,
                                        maxWidth: 500,
                                        fontWeight: "bold",
                                        fontSize: "12px",
                                        textAlign: "center",
                                        color: "#FF0000",
                                    }}
                                >
                                    {" "}
                                    Positive Content Score is a float in the
                                    range [0, 1].
                                </Typography>
                                <Typography
                                    sx={{
                                        p: 1,
                                        maxWidth: 500,
                                        fontWeight: "bold",
                                        fontSize: "12px",
                                        textAlign: "center",
                                        color: "#FF0000",
                                    }}
                                >
                                    0 = no positive content, 1 = all positive
                                    content
                                </Typography>
                                <hr className="border-t-3 border-gray-300 my-4 ml-5 mr-5"></hr>
                                <Typography
                                    sx={{
                                        p: 1,
                                        maxWidth: 500,
                                        textAlign: "center",
                                        fontWeight: "bold",
                                    }}
                                >
                                    How's Positive Content Score calculated?
                                </Typography>
                                <Typography sx={{ p: 2, maxWidth: 500 }}>
                                    400 top posts of all time from r/{name} are
                                    sampled. Only the TITLE and DESCRIPTION of
                                    the posts are analyzed.
                                </Typography>
                                <Typography sx={{ p: 2, maxWidth: 500 }}>
                                    The python class
                                    'nltk.sentiment.vader.SentimentIntensityAnalyzer'
                                    analyzes all the post text and scores the
                                    text from -1 to 1 (with -1 being most
                                    negative, 0 being neutral, and 1 being most
                                    positive). This score is normalized to [0,
                                    1].
                                </Typography>
                                <Typography sx={{ p: 2, maxWidth: 500 }}>
                                    Posts that have more upvotes have a greater
                                    effect on the overall positive content score
                                    (i.e. the positive content score is a
                                    weighted average)
                                </Typography>
                            </ButtonPopover>
            );
        };

        const GradeCircle: React.FC<{ grade: string }> = ({ grade }) => {
            const getGradeBackgroundColor = () => {
                if (["A+", "A", "A-"].includes(grade)) {
                    return "bg-green-300";
                }
                if (["B+", "B", "B-"].includes(grade)) {
                    return "bg-lime-300";
                }
                if (["C+", "C", "C-"].includes(grade)) {
                    return "bg-yellow-300";
                }
                if (["D+", "D", "D-"].includes(grade)) {
                    return "bg-orange-300";
                }
                if (["F"].includes(grade)) {
                    return "bg-red-300";
                }
            };

            const getGradeBorderColor = () => {
                if (["A+", "A", "A-"].includes(grade)) {
                    return "border-4 border-green-400";
                }
                if (["B+", "B", "B-"].includes(grade)) {
                    return "border-4 border-lime-400";
                }
                if (["C+", "C", "C-"].includes(grade)) {
                    return "border-4 border-yellow-400";
                }
                if (["D+", "D", "D-"].includes(grade)) {
                    return "border-4 border-orange-400";
                }
                if (["F"].includes(grade)) {
                    return "border-4 border-red-500";
                }
            };

            return (
                <span
                    className={`inline-flex items-center justify-center w-7 h-7 ${getGradeBackgroundColor()} 
                    ${getGradeBorderColor()} text-black text-[60%] m-3 rounded-full`}
                >
                    {grade}
                </span>
            );
        };

        const Histogram: React.FC<{
            values: number[];
            xaxis_label: string;
        }> = ({ values, xaxis_label }) => {
            //console.log(values)
            const histogramData = createHistogram(values);

            return (
                <div className="mt-5 h-60 w-[90%] p-4 bg-white shadow-md rounded-xl">
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                            data={histogramData}
                            margin={{ bottom: 15, left: 7 }}
                        >
                            <XAxis
                                dataKey="bin"
                                tick={{ fontSize: 10, fill: "black" }}
                                label={{
                                    value: xaxis_label,
                                    fill: "black",
                                    position: "outsideBottom",
                                    dy: 20,
                                    dx: -12
                                }}
                            />
                            <YAxis
                                tick={{ fill: "black" }}
                                label={{
                                    value: "Count",
                                    fill: "black",
                                    angle: -90,
                                    position: "insideLeft",
                                    dy: 20,
                                    dx: -4,
                                }}
                            />
                            <Tooltip />
                            <Bar dataKey="count" fill="#4f46e5"></Bar>
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            );
        };

        const GradeBarGraph: React.FC<{ all_scores: string[] }> = ({
            all_scores,
        }) => {

            const grade_to_count = all_scores.reduce(
                (acc: Record<string, number>, grade: string) => {
                    acc[grade] = (acc[grade] || 0) + 1; // Increment count or initialize to 1
                    return acc;
                },
                {}
            );
            const gradeData = Object.entries(grade_to_count).map(
                ([label, count]) => ({
                    label,
                    count,
                })
            );
            const gradeOrder = [
                "A+",
                "A",
                "A-",
                "B+",
                "B",
                "B-",
                "C+",
                "C",
                "C-",
                "D+",
                "D",
                "D-",
                "F",
            ];
            gradeData.sort(
                (a, b) =>
                    gradeOrder.indexOf(a.label) - gradeOrder.indexOf(b.label)
            );

            function getBarColor(grade: string) {
                if (["A+", "A", "A-"].includes(grade)) return "#00A36C"; // Green
                if (["B+", "B", "B-"].includes(grade)) return "#a3e635"; // Lime
                if (["C+", "C", "C-"].includes(grade)) return "#FDDA0D"; // Yellow
                if (["D+", "D", "D-"].includes(grade)) return "#fb923c"; // Orange
                if (grade === "F") return "#EE4B2B"; // Red
            }

            return (
                <div className="mt-2 mr-10 mb-8 h-50 w-[90%] bg-white shadow-md rounded-xl">
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={gradeData}>
                            <XAxis
                                dataKey="label"
                                tick={{ fill: "black", fontSize: 12 }}
                            />
                            <YAxis
                                tick={{ fill: "black" }}
                                label={{
                                    value: "Count",
                                    angle: -90,
                                    position: "insideLeft",
                                    fill: "black",
                                }}
                            />
                            <Tooltip />
                            <Bar dataKey="count">
                                {gradeData.map((entry, index) => (
                                    <Cell
                                        key={`cell-${index}`}
                                        fill={getBarColor(entry.label)}
                                    />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            );
        };

        return (
            <div className="mt-15 ml-8">
                <div className="flex flex-col items-center justify-center">
                    <div className="w-1/2">
                        <div className="text-center">
                            <h1 className="text-[18px] font-bold">
                                Ranking Toxicity Score
                            </h1>
                            <br/>
                            <ButtonPopover title="What's Toxicity Score?">
                            <Typography
                                    sx={{
                                        p: 2,
                                        maxWidth: 500,
                                        fontWeight: "bold",
                                        fontSize: "18px",
                                        textAlign: "center",
                                        color: "#FF0000",
                                    }}
                                >
                                    {" "}
                                    Toxicity Score is a float in the range [0,
                                    1].
                                </Typography>
                                <Typography
                                    sx={{
                                        p: 1,
                                        maxWidth: 500,
                                        fontWeight: "bold",
                                        fontSize: "12px",
                                        textAlign: "center",
                                        color: "#FF0000",
                                    }}
                                >
                                    0 = no toxic content, 1 = all toxic content
                                </Typography>
                                <hr className="border-t-3 border-gray-300 my-4 ml-5 mr-5"></hr>
                                <Typography
                                    sx={{
                                        p: 1,
                                        maxWidth: 500,
                                        textAlign: "center",
                                        fontWeight: "bold",
                                    }}
                                >
                                    How is Toxicity Score calculated?
                                </Typography>
                                <Typography sx={{ p: 1, maxWidth: 500 }}>
                                    400 top posts of all time from r/{name} are
                                    sampled. Only the TITLE and DESCRIPTION of
                                    the posts are analyzed.
                                </Typography>
                                <Typography sx={{ p: 1, maxWidth: 500 }}>
                                    The python library 'Detoxify' analyzes all
                                    the post text and scores the text on 6
                                    metrics: toxicity, severe_toxicity, obscene,
                                    threat, insult, identify attack. A weighted
                                    average of these 6 metrics is taken to get
                                    an overall toxicity score.
                                </Typography>
                                <Typography
                                    sx={{
                                        p: 1,
                                        maxWidth: 500,
                                        fontWeight: "bold",
                                    }}
                                >
                                    NOTE: 'Detoxify' analyzes text LITERALLY. It
                                    cannot grasp humor or sarcasm (so dark jokes
                                    will have a high toxic score)
                                </Typography>
                            </ButtonPopover>
                            <br/>
                        </div>
                        <ul className="list-disc">
                            <li>
                                <h1 className="text-base font-semibold m-1">
                                    toxicity score ={" "}
                                    {Number(analysis.toxicity_score.toFixed(2))}
                                </h1>
                            </li>
                            <li>
                                <h1 className="text-base font-semibold m-1">
                                    toxicity grade =
                                    <GradeCircle
                                        grade={analysis.toxicity_grade}
                                    ></GradeCircle>
                                </h1>
                                <GradeBarGraph
                                    all_scores={analysis.all_toxicity_grades}
                                ></GradeBarGraph>
                            </li>
                            <li>
                                <h1 className="text-base font-semibold m-1">
                                    toxicity percentile ={" "}
                                    {Number(
                                        analysis.toxicity_percentile.toFixed(1)
                                    )}
                                    %
                                </h1>
                                <h1 className="text-base text-[15px] max-w-110 m-1 italic">
                                    This means that r/{name} has a higher
                                    toxicity score than{" "}
                                    {Number(
                                        analysis.toxicity_percentile.toFixed(1)
                                    )}
                                    % of the 4000+ sampled subreddits{" "}
                                </h1>
                            </li>
                        </ul>
                        <Histogram
                            values={analysis.all_toxicity_scores}
                            xaxis_label="Toxicity Score"
                        ></Histogram>
                    </div>
                    <br></br>
                    <br></br>
                    <hr className="bg-gray-400 border-2 w-full"></hr>
                    <br></br>
                    <div className="w-1/2">
                    <div className="w-[1px] bg-gray-400"></div>
                    <div className="items-center">
                        <div className="text-center">
                            <h1 className="text-[18px] font-bold">
                                Ranking Positive Content Score
                            </h1>
                            <br/>
                            <PositiveContentPopoverButton></PositiveContentPopoverButton>
                            <br/>
                        </div>
                        <ul className="list-disc">
                            <li>
                                <h1 className="text-[16px] font-semibold m-1">
                                    positive content score ={" "}
                                    {Number(
                                        analysis.positive_content_score.toFixed(
                                            2
                                        )
                                    )}
                                </h1>
                            </li>
                            <li>
                                <h1 className="text-[16px] font-semibold m-1">
                                    positive content grade =
                                    <GradeCircle
                                        grade={analysis.positive_content_grade}
                                    ></GradeCircle>
                                </h1>
                                <GradeBarGraph
                                    all_scores={
                                        analysis.all_positive_content_grades
                                    }
                                ></GradeBarGraph>
                            </li>
                            <li>
                                <h1 className="text-[16px] font-semibold m-1">
                                    positive content percentile ={" "}
                                    {Number(
                                        analysis.positive_content_percentile.toFixed(
                                            1
                                        )
                                    )}
                                    %
                                </h1>
                                <h1 className="text-[15px] max-w-110 m-1 italic">
                                    This means that r/{name} has a higher
                                    positive content score than{" "}
                                    {Number(
                                        analysis.positive_content_percentile.toFixed(
                                            1
                                        )
                                    )}
                                    % of the 4000+ sampled subreddits
                                </h1>
                            </li>
                        </ul>
                        <Histogram
                            values={analysis.all_positive_content_scores}
                            xaxis_label="Positive Content Score"
                        ></Histogram>
                    </div>
                </div>
                </div>
            </div>
        );
    };

    const renderNGrams = () => {
        if (!analysis) return null;

        const NGramsForDate: React.FC<{ date: string; ngrams: NGram[] }> = ({
            date,
            ngrams,
        }) => {
            return (
                <div
                    key={date}
                    className="flex flex-col h-full mb-3 mr-3 ml-3 border bg-white border-gray-200 rounded-l shadow-md"
                >
                    <h3 className="text-lg text-center p-1 font-semibold bg-gray-100">
                        {formatDate(date, timeFilter)}
                    </h3>
                    <div className="flex font-semibold pt-1 pb-1 pr-5 pl-5 bg-indigo-200 justify-between">
                        <h2>Word</h2>
                        <h2>Count</h2>
                    </div>
                    <ul className="list-none pl-5 pr-5 pt-2">
                        {ngrams.map((ngram: NGram, index: number) => (
                            <li key={index} className="pb-1 pt-1">
                                <div className="text-[14px] flex justify-between">
                                    <div>{ngram[0]}</div>
                                    <div>{ngram[1]}</div>
                                </div>
                                <hr className="border-t border-gray-300 my-1" />
                            </li>
                        ))}
                    </ul>
                </div>
            );
        };

        return (
            <div className="mt-7">
                <div className="flex flex-col items-center">
                    <h2 className="text-xl text-center font-bold">
                        Most Mentioned Bi-Grams
                    </h2>
                    <br/>
                    <ButtonPopover title="What's a Bi-Gram?">
                        A Bi-Gram is 2 consecutive words in a text.
                        For example, the sentence "I love pizza" has
                        2 bigrams: "I love" and "love pizza".
                    </ButtonPopover>
                </div>
                <div
                    style={{
                        marginTop: "0px",
                        marginBottom: "40px",
                        marginLeft: "25px",
                        marginRight: "25px",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        position: "relative",
                    }}
                >
                </div>
                 <div className="grid grid-cols-4">
                        {Object.entries(analysis.top_n_grams).map(
                            ([date, ngrams]) => ngrams.length > 0 && (
                                <div
                                    className="mb-4"
                                    key={date}
                                >
                                    <NGramsForDate
                                        date={date}
                                        ngrams={ngrams}
                                    />
                                </div>
                            )
                        )}
                    </div>
              </div>
            // </div>
        );
    };

    const renderNamedEntities = () => {
        if (!analysis) return null;

        /* STILL A WORK IN PROGRESS */
        // const RenderConsistentlyMentionedNamedEntities = () => {
        //     if (!analysis) return null;
        //     const entities = analysis.top_named_entities
        //     const all_entities: string[] = Object.values(entities).flatMap((entityArray) => entityArray.map((entity) => entity[0]));
        //     const frequencyMap: Record<string, number> = {};
        //     all_entities.forEach((text) => {frequencyMap[text] = (frequencyMap[text] || 0) + 1;});
        //     const consistently_mentioned_entities: string[] = Object.keys(frequencyMap)
        //         .filter((text) => frequencyMap[text] >= 3);
        //     return (
        //         <div className="self-start w-auto border border-1 shadow rounded mt-10 ml-4 text-left">
        //             <h1 className="font-semibold text-[16px] p-2 pl-6 pr-6">r/{name} consistently talked about these entities</h1>
        //             <div className="mt-3 mb-3">
        //                 {consistently_mentioned_entities.map((entity, idx) => (<h1 className="mb-2 ml-6" key={idx}>{entity}</h1>))}
        //             </div>
        //         </div>
        //     );
        // };

        const ColorCodeBox = (props: any) => {
            return (
                <div
                    className="w-auto font-medium border-1"
                    style={{
                        fontSize: "75%",
                        backgroundColor: props.backgroundColor,
                        textAlign: "center",
                        paddingLeft: "10px",
                        paddingRight: "10px",
                        paddingBottom: "7px",
                        paddingTop: "7px",
                        maxWidth: "160px", 
                        borderTopLeftRadius: props.borderTopLeftRadius,
                        borderBottomLeftRadius: props.borderBottomLeftRadius,
                        borderTopRightRadius: props.borderTopRightRadius,
                        borderBottomRightRadius: props.borderBottomRightRadius,
                    }}
                >
                    {props.label}
                </div>
            );
        };

        const TopNamedEntitiesForDate = (props: any) => {
            const SummarizedSentimentBulletPoints: React.FC<{
                summary: string;
            }> = ({ summary }) => {
                if (!summary) return null;
                const sentences = summary
                    .split("---")
                    .filter((sentence: string) => sentence.trim() !== "");
                return (
                    <ul className="list-disc pl-4">
                        {" "}
                        {sentences.map((sentence: string, index: number) => (
                            <li key={index}>{sentence.trim()}</li>
                        ))}
                    </ul>
                );
            };

            const LinksForEntity = ({entity}: { entity: NamedEntity}) => {
                if(entity[4].length == 0) { return; }
                return (
                    <div className="mt-10 shadow rounded p-3 bg-gray-100">
                        <h1 className="font-semibold mb-4 text-[14px]">Top Reddit Post discussing "{entity[0]}":</h1>
                        {Object.entries(entity[4]).map(([_, link]) => (
                            // <div key={link}>post: {link}</div>
                            <img 
                                src={"/link_icon.png"}
                                className="rounded-full bg-indigo-200 p-2 w-11 h-11 object-cover cursor-pointer transition active:scale-95 active:brightness-90"
                                onClick={() => window.open(link, "_blank")}
                            />
                        ))}
                    </div>
                );
            };

            return (
                <div
                    key={props.date}
                    className="flex-grow flex-1 mb-6 mr-1 ml-1 border bg-white border-gray-300 rounded-l shadow-md"
                >
                    <h3
                        className="text-lg font-semibold bg-gray-100"
                        style={{
                            fontSize: "20px",
                            textAlign: "center",
                            padding: "4px",
                        }}
                    >
                        {formatDate(props.date, timeFilter)}
                    </h3>
                    <div className="grid grid-cols-3 pl-0">
                        {props.entities.map(
                            (entity: NamedEntity, index: number) => {
                                const backgroundColor =
                                    entity[2] >= 0.5 && entity[2] <= 1
                                        ? "#AFE1AF"
                                        : entity[2] >= 0.1 && entity[2] < 0.5
                                        ? "#e2f4a5"
                                        : entity[2] >= -0.1 && entity[2] < 0.1
                                        ? "#FFFFC5"
                                        : entity[2] >= -0.5 && entity[2] < -0.1
                                        ? "#FFD580"
                                        : entity[2] >= -1 && entity[2] < -0.5
                                        ? "#ffb9b9"
                                        : "bg-gray-100";
                                return (
                                    <div key={index} className="border border-gray-300 p-4 shadow-sm bg-white">
                                        <div className="flex justify-center">
                                            <div
                                                className="inline-block justify-center items-center text-center mb-2 font-[14.5px] font-semibold pl-3 pr-3 pt-1 pb-1"
                                                style={{
                                                    backgroundColor:
                                                        backgroundColor,
                                                }}
                                            >
                                                {entity[0]}
                                            </div>
                                        </div>
                                        <div className="ml-5 mr-5 mt-3 mb-3">
                                            <div
                                                style={{
                                                    display: "flex",
                                                    justifyContent:
                                                        "space-between",
                                                    alignItems: "center",
                                                }}
                                            >
                                                <span
                                                    style={{
                                                        fontWeight: 600,
                                                        fontSize: "14px",
                                                    }}
                                                >
                                                    Sentiment Score [-1, 1]:{" "}
                                                    {entity[2]}
                                                </span>
                                                <span
                                                    style={{
                                                        fontWeight: 600,
                                                        fontSize: "14px",
                                                    }}
                                                >
                                                    # Mentions: {entity[1]}
                                                </span>
                                            </div>

                                            <div
                                                style={{
                                                    fontWeight: 600,
                                                    fontSize: "14px",
                                                    marginTop: "10px",
                                                    marginBottom: "5px",
                                                }}
                                            >
                                                Key Points:{" "}
                                            </div>
                                            <div
                                                style={{
                                                    fontSize: "13.5px",
                                                    color: "#333",
                                                }}
                                            >
                                                <SummarizedSentimentBulletPoints
                                                    summary={entity[3]}
                                                ></SummarizedSentimentBulletPoints>
                                                 <LinksForEntity entity={entity}></LinksForEntity>
                                            </div>
                                        </div>
                                    </div>
                                );
                            }
                        )}
                    </div>
                </div>
            );
        };

        /* For switching between different dates quickly */
        const NamedEntitiesMenu = ({dates}: { dates: string[]}) => {
            return (
                    <div className="flex mt-2 mb-6 w-auto bg-white border border-2 border-gray-300 shadow-lg z-10">
                      {Object.entries(dates).map(
                            ([idx, date]) => (
                                <div
                                    key={date}
                                    className={Number(idx) !== 0 ? "border-l-2 border-gray-300" : ""}
                                >
                                   <div onClick={() => setCurrentNamedEntityDate(date)}  className={`px-4 py-2 cursor-pointer ${currentNamedEntityDate === date ? "bg-indigo-100 text-black" : ""}`}>{formatDate(date, timeFilter)}</div>
                                </div>
                            )
                        )}
                    </div>
              );
        };

        const CurrentNamedEntity = () => {
            if (!analysis) {
                return;
            }

            if (currentNamedEntityDate == "") {
                setCurrentNamedEntityDate(Object.keys(analysis.top_named_entities)[0])
            }

            return (
                <TopNamedEntitiesForDate date={currentNamedEntityDate} entities={analysis.top_named_entities[currentNamedEntityDate]}></TopNamedEntitiesForDate>
            );
        };

        return (
            <div className="mt-6">
                <h2
                        className="text-xl font-bold text-center"
                        style={{ textAlign: "center", fontSize: "20px" }}
                    >
                        Most Mentioned Named Entities
                    </h2>
                <div className="flex relative items-center text-center justify-center">
                    <br/>
                    <div className="flex justify-center m-8">
                        <ColorCodeBox
                            backgroundColor="#AFE1AF"
                            label="Very Positive Consensus"
                        ></ColorCodeBox>
                        <ColorCodeBox
                            backgroundColor="#e2f4a5"
                            label="Positive Consensus"
                        ></ColorCodeBox>
                        <ColorCodeBox
                            backgroundColor="#FFFFC5"
                            label="Neutral Consensus"
                        ></ColorCodeBox>
                        <ColorCodeBox
                            backgroundColor="#FFD580"
                            label="Negative Consensus"
                        ></ColorCodeBox>
                        <ColorCodeBox
                            backgroundColor="#ffb9b9"
                            label="Very Negative Consensus"
                        ></ColorCodeBox>
                    </div>
                </div>
                <div className="flex flex-col items-center justify-center">
                    <NamedEntitiesMenu dates={Array.from(Object.keys(analysis.top_named_entities))}></NamedEntitiesMenu>
                    <CurrentNamedEntity></CurrentNamedEntity>
                </div>
                <div className="flex gap-5 mb-5 mt-5 items-center justify-center">
                            <ButtonPopover title="What's a named entity?">
                                A Named Entity is a key subject in a piece
                                of text (include names of people,
                                organizations, locations, and dates)
                            </ButtonPopover>
                            <ButtonPopover title="How's sentiment score computed?">
                                The aspect-based sentiment analysis model
                                <strong>
                                    "yangheng/deberta-v3-base-absa-v1.1"
                                </strong>
                                is used to generate a sentiment score from
                                -1 (most negative) to 1 (most positive).
                                This sentiment score represents how the
                                writer feels TOWARD the named entity (and is
                                independent of the sentence's overall tone &
                                sentiment).
                                <br/>
                                <hr/>
                                <br/>
                                Example:
                                <i>
                                    "It's so unfair that Taylor Swift didn't get
                                    nominated for a grammy this year - she got
                                    cheated"
                                </i>
                                    The overall tone/sentiment of this sentence
                                    is frustrated/angry, but the writer feels <strong style={{ color: "green" }}> POSITIVELY </strong> 
                                    toward Taylor Swift because they believe she
                                    is deserving of a grammy.
                            </ButtonPopover>
                            <ButtonPopover title='How are "Key Points" computed?'>
                                The summarization ML model "facebook/bart-large-cnn" processes all of the sentences where a 
                                named entity is directly mentioned and creates a short summary.
                            </ButtonPopover>
                    </div>
            </div>
        );
    };

    const spinnerStyle = "animate-spin h-6 w-6 border-t-4 border-blue-500 border-solid rounded-full";

    const Menu = () => {
        return (
            <div className="relative ml-7 inline-block text-left mb-60">
              <button className="font-semibold bg-indigo-600 text-white px-4 py-2 rounded">
                Menu
              </button>
                <div className="absolute mt-2 w-45 bg-white border border-gray-200 rounded shadow-lg z-10">
                  <ul className="py-1">
                    {timeFilter == "all" && <li className={`px-4 py-2 cursor-pointer ${currentMenuItem === "Ranking" ? "bg-indigo-100 text-black" : ""}`} onClick={() => setCurrentMenuItem("Ranking")}>Ranking</li>}
                    {timeFilter == "all" && <hr></hr>}
                    <li className={`px-4 py-2 cursor-pointer ${currentMenuItem === "Named Entities" ? "bg-indigo-100 text-black" : ""}`} onClick={() => setCurrentMenuItem("Named Entities")}>Named Entities</li>
                    <hr></hr>
                    <li className={`px-4 py-2 cursor-pointer ${currentMenuItem === "Bi-Grams" ? "bg-indigo-100 text-black" : ""}`}  onClick={() => setCurrentMenuItem("Bi-Grams")}>Bi-Grams</li>
                    <hr></hr>
                    <li className={`px-4 py-2 cursor-pointer ${currentMenuItem === "Word Embeddings" ? "bg-indigo-100 text-black" : ""}`}  onClick={() => setCurrentMenuItem("Word Embeddings")}>Word Embeddings</li>
                    <hr></hr>
                    <li className={`px-4 py-2 cursor-pointer ${currentMenuItem === "Word Cloud" ? "bg-indigo-100 text-black" : ""}`} onClick={() => setCurrentMenuItem("Word Cloud")}>Word Cloud</li>
                    <hr></hr>
                    <li className={`px-4 py-2 cursor-pointer ${currentMenuItem === "Readability Metrics" ? "bg-indigo-100 text-black" : ""}`}  onClick={() => setCurrentMenuItem("Readability Metrics")}>Readability Metrics</li>
                  </ul>
                </div>
            </div>
          );
    };

    const DisplayedAnalysis = () => {
        console.log("currentMenuItem: ", currentMenuItem)
        return (
        <div>
            {timeFilter == "all" && currentMenuItem == "Ranking" && renderComparativeAnalysis()}
            {currentMenuItem == "Bi-Grams" && renderNGrams()}
            {currentMenuItem == "Named Entities" && renderNamedEntities()}
            {currentMenuItem == "Word Embeddings" && renderWordEmbeddings()}
            {currentMenuItem == "Word Cloud" && renderWordCloud()}
            {currentMenuItem == "Readability Metrics" && renderReadabilityMetrics()}
        </div>
        );
    };

    return (
        <div>
            <div className="ml-2 mr-2 mt-4 mb-4 p-5 bg-white rounded-md shadow-sm">
                <div className="flex flex-col items-center mb-6">
                    <SubredditAvatar subredditName={name ?? ""} />
                    <h1 className="text-lg font-bold">r/{name}</h1>
                </div>
                <div className="flex justify-center items-center gap-4 mb-2">
                    <div>
                        <label className="block text-sm text-center font-medium text-gray-700"> Time Filter </label>
                        <select
                            value={timeFilter} onChange={(e) => { setTimeFilter(e.target.value); }}
                            className="mt-1 block w-full px-1 py-2 text-base border border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                        >
                            <option value="all">All Time</option>
                            <option value="year">Year</option>
                            <option value="week">Week</option>
                        </select>
                    </div>
                    <div>
                        <label className="block text-sm text-center font-medium text-gray-700"> Sort By </label>
                        <select
                            value={sortBy} onChange={(e) => setSortBy(e.target.value)}
                            className="mt-1 block w-full px-1 py-2 text-base border border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                        >
                            <option value="top">Top Posts</option>          
                        </select>
                    </div>
                    <div className="relative flex gap-4">
                        {isBusy && !analysis && <div className={`absolute ${spinnerStyle} right-[-40px]`}></div>}
                    </div>
                </div>

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

                {!error && analysis && (
                    <div className="mt-4">
                        <hr className="my-4 border-t border-gray-400 mx-auto w-[97%]" />
                        <div className="m-4 pl-4 text-[15px]">
                            <div className="flex gap-2 justify-center">
                                <h1 className="text-white bg-orange-500 font-bold w-7 h-7 rounded-full text-center text-[18px]">!</h1>
                                <h1 className="mb-4 mt-1 text-orange-600 font-semibold text-center">This analysis may be outdated</h1>
                            </div>
                            <h1 className="mb-4">analysis generated on: <span className="bg-orange-200 font-bold p-1 rounded-sm">{new Date(analysis.timestamp * 1000).toLocaleString()}</span>
                            </h1>
                            <h1 className="mb-4">analysis analyzed <span className="bg-orange-200 font-bold p-1 rounded-sm">{analysis.num_words}</span> words</h1>  
                        </div>
                        <hr className="my-4 border-t border-gray-400 mx-auto w-[97%]" />
                        <div className="flex flex-col">
                            <Menu></Menu>
                            <DisplayedAnalysis></DisplayedAnalysis>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
