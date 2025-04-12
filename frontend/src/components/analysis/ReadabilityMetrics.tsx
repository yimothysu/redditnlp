import { SubredditAnalysis } from "../../types/redditAnalysis";
import ButtonPopover from "../ButtonPopover";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { Metric } from "./Metric";
interface ReadabilityMetricsProps {
  analysis: SubredditAnalysis;
}

export const ReadabilityMetrics: React.FC<ReadabilityMetricsProps> = ({
  analysis,
}) => {
  
  const data = Object.entries(analysis.readability_metrics.dale_chall_grade_levels).map(([key, value]) => ({
    bin: key,
    count: value
  }))

  const histogram = analysis.readability_metrics["dale_chall_grade_levels"] && Object.keys(analysis.readability_metrics["dale_chall_grade_levels"]).length > 0 ? 
    <div className="mt-5 flex justify-content align-center h-45 sm:h-80 w-[100%] p-1 bg-white rounded-xl">
    <ResponsiveContainer width="100%" height="100%">
        <BarChart
            data={data}
            margin={{ bottom: 15, left: 7 }}
        >
            <XAxis
                dataKey="bin"
                tick={{ fontSize: 10, fill: "black" }}
                label={{
                    value: "Grade Level",
                    fill: "black",
                    position: "outsideBottom",
                    dy: 20,
                    dx: -12
                }}
            />
            <YAxis
                tick={{ fill: "black" }}
                label={{
                    value: "# Posts",
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
</div> : <h1 className="text-center text-gray-400">No data to provide</h1>;

  return (
    <div>
      <h1 className="font-bold text-xl text-center mt-5 p-1">
        Readability Metrics
      </h1>
      <br />
      <div className="text-center">
        <ButtonPopover title="What are readability metrics">
          Readability metrics evaluate how easy a body text is to read. This
          includes expected grade level of education required to understand a
          text. To learn more about Flesch-Kincaid scores and Dale Chall metrics, see <a href="https://pypi.org/project/py-readability-metrics/#flesch-kincaid-grade-level">here</a>.
          Note: Grade level data can only be calculated for posts of 100 words or greater.
        </ButtonPopover>
        </div>
      <br />
      <div className="flex flex-col align-center items-center">
        <Metric name="Avg Word Count Per Title" metric={analysis.readability_metrics["avg_num_words_title"]}/>
        <Metric name="Avg Word Count Per Description" metric={analysis.readability_metrics["avg_num_words_description"]}/>
        <Metric name="Avg Flesch Grade Level" metric={analysis.readability_metrics["avg_flesch_grade_level"] == -1 ? "No Data Available": analysis.readability_metrics["avg_flesch_grade_level"]}/>
        <div className="border bg-white border-gray-200 rounded-lg shadow-md m-2 sm:w-7/8 p-3">
          <p className="text-center text-gray-500">
            <b>Distribution of Grade Levels (Dale Chall)</b>
            {histogram}
          </p>
        </div>
      </div>
    </div>
  );
};