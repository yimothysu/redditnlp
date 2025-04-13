import { SubredditAnalysis } from "../../types/redditAnalysis";
import ButtonPopover from "../ButtonPopover";
import { GradeCircle } from "./GradeCircle";
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
import { createHistogram } from "../../utils/dateUtils";

interface ComparativeAnalysisProps {
  analysis: SubredditAnalysis;
}

interface HistogramProps {
  values: number[];
  xaxis_label: string;
}

interface GradeBarGraphProps {
  all_scores: string[];
}

const PositiveContentPopoverButton = ({ name }: { name: string }) => {
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
        Positive Content Score is a float in the range [0, 1].
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
        0 = no positive content, 1 = all positive content
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
        400 top posts of all time from r/{name} are sampled. Only the TITLE and
        DESCRIPTION of the posts are analyzed.
      </Typography>
      <Typography sx={{ p: 2, maxWidth: 500 }}>
        The python class 'nltk.sentiment.vader.SentimentIntensityAnalyzer'
        analyzes all the post text and scores the text from -1 to 1 (with -1
        being most negative, 0 being neutral, and 1 being most positive). This
        score is normalized to [0, 1].
      </Typography>
      <Typography sx={{ p: 2, maxWidth: 500 }}>
        Posts that have more upvotes have a greater effect on the overall
        positive content score (i.e. the positive content score is a weighted
        average)
      </Typography>
    </ButtonPopover>
  );
};

const Histogram: React.FC<HistogramProps> = ({ values, xaxis_label }) => {
  const histogramData = createHistogram(values);

  return (
    <div className="mt-5 h-60 w-[90%] p-4 bg-white shadow-md rounded-xl">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={histogramData} margin={{ bottom: 15, left: 7 }}>
          <XAxis
            dataKey="bin"
            tick={{ fontSize: 10, fill: "black" }}
            label={{
              value: xaxis_label,
              fill: "black",
              position: "outsideBottom",
              dy: 20,
              dx: -12,
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

const GradeBarGraph: React.FC<GradeBarGraphProps> = ({ all_scores }) => {
  const grade_to_count = all_scores.reduce(
    (acc: Record<string, number>, grade: string) => {
      acc[grade] = (acc[grade] || 0) + 1;
      return acc;
    },
    {}
  );
  const gradeData = Object.entries(grade_to_count).map(([label, count]) => ({
    label,
    count,
  }));
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
    (a, b) => gradeOrder.indexOf(a.label) - gradeOrder.indexOf(b.label)
  );

  function getBarColor(grade: string) {
    if (["A+", "A", "A-"].includes(grade)) return "#00A36C"; // Green
    if (["B+", "B", "B-"].includes(grade)) return "#a3e635"; // Lime
    if (["C+", "C", "C-"].includes(grade)) return "#FDDA0D"; // Yellow
    if (["D+", "D", "D-"].includes(grade)) return "#fb923c"; // Orange
    if (grade === "F") return "#EE4B2B"; // Red
    return "#000000"; // Default black
  }

  return (
    <div className="mt-2 mr-10 mb-8 h-50 w-[90%] bg-white shadow-md rounded-xl">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={gradeData}>
          <XAxis dataKey="label" tick={{ fill: "black", fontSize: 12 }} />
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
              <Cell key={`cell-${index}`} fill={getBarColor(entry.label)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export const ComparativeAnalysis: React.FC<ComparativeAnalysisProps> = ({
  analysis,
}) => {
  return (
    <div className="mt-15 ml-8">
      <div className="flex flex-col items-center justify-center">
        <div className="w-1/2">
          <div className="text-center">
            <h1 className="text-[18px] font-bold">Ranking Toxicity Score</h1>
            <br />
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
                Toxicity Score is a float in the range [0, 1].
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
                400 top posts of all time are sampled. Only the TITLE and
                DESCRIPTION of the posts are analyzed.
              </Typography>
              <Typography sx={{ p: 1, maxWidth: 500 }}>
                The python library 'Detoxify' analyzes all the post text and
                scores the text on 6 metrics: toxicity, severe_toxicity,
                obscene, threat, insult, identify attack. A weighted average of
                these 6 metrics is taken to get an overall toxicity score.
              </Typography>
              <Typography
                sx={{
                  p: 1,
                  maxWidth: 500,
                  fontWeight: "bold",
                }}
              >
                NOTE: 'Detoxify' analyzes text LITERALLY. It cannot grasp humor
                or sarcasm (so dark jokes will have a high toxic score)
              </Typography>
            </ButtonPopover>
            <br />
          </div>
          <ul className="list-disc">
            <li>
              <h1 className="text-base font-semibold m-1">
                toxicity score = {Number(analysis.toxicity_score.toFixed(2))}
              </h1>
            </li>
            <li>
              <h1 className="text-base font-semibold m-1">
                toxicity grade =
                <GradeCircle grade={analysis.toxicity_grade} />
              </h1>
              <GradeBarGraph all_scores={analysis.all_toxicity_grades} />
            </li>
            <li>
              <h1 className="text-base font-semibold m-1">
                toxicity percentile ={" "}
                {Number(analysis.toxicity_percentile.toFixed(1))}%
              </h1>
              <h1 className="text-base text-[15px] max-w-110 m-1 italic">
                This means that this subreddit has a higher toxicity score than{" "}
                {Number(analysis.toxicity_percentile.toFixed(1))}% of the 4000+
                sampled subreddits{" "}
              </h1>
            </li>
          </ul>
          <Histogram
            values={analysis.all_toxicity_scores}
            xaxis_label="Toxicity Score"
          />
        </div>
        <br />
        <br />
        <hr className="bg-gray-400 border-2 w-full"></hr>
        <br />
        <div className="w-1/2">
          <div className="w-[1px] bg-gray-400"></div>
          <div className="items-center">
            <div className="text-center">
              <h1 className="text-[18px] font-bold">
                Ranking Positive Content Score
              </h1>
              <br />
              <PositiveContentPopoverButton name="subreddit" />
              <br />
            </div>
            <ul className="list-disc">
              <li>
                <h1 className="text-[16px] font-semibold m-1">
                  positive content score ={" "}
                  {Number(analysis.positive_content_score.toFixed(2))}
                </h1>
              </li>
              <li>
                <h1 className="text-[16px] font-semibold m-1">
                  positive content grade =
                  <GradeCircle grade={analysis.positive_content_grade} />
                </h1>
                <GradeBarGraph
                  all_scores={analysis.all_positive_content_grades}
                />
              </li>
              <li>
                <h1 className="text-[16px] font-semibold m-1">
                  positive content percentile ={" "}
                  {Number(analysis.positive_content_percentile.toFixed(1))}%
                </h1>
                <h1 className="text-[15px] max-w-110 m-1 italic">
                  This means that this subreddit has a higher positive content
                  score than{" "}
                  {Number(analysis.positive_content_percentile.toFixed(1))}% of
                  the 4000+ sampled subreddits
                </h1>
              </li>
            </ul>
            <Histogram
              values={analysis.all_positive_content_scores}
              xaxis_label="Positive Content Score"
            />
          </div>
        </div>
      </div>
    </div>
  );
};
