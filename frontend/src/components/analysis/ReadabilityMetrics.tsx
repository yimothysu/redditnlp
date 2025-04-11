import { SubredditAnalysis } from "../../types/redditAnalysis";
import ButtonPopover from "../ButtonPopover";

interface ReadabilityMetricsProps {
  analysis: SubredditAnalysis;
}

export const ReadabilityMetrics: React.FC<ReadabilityMetricsProps> = ({
  analysis,
}) => {
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
          text.
        </ButtonPopover>
      </div>
      <br />
      <p className="text-center text-gray-800">
        <b>Average Number of Words Per Post Title: </b>
        {analysis.readability_metrics["avg_num_words_title"]}
      </p>
      <p className="text-center text-gray-800">
        <b>Average Number of Words Per Post Description: </b>
        {analysis.readability_metrics["avg_num_words_description"]}
      </p>
      <p className="text-center text-gray-800">
        <b>Average Grade Level of Text (Flesch Score): </b>
        {analysis.readability_metrics["avg_flesch_grade_level"] != -1
          ? analysis.readability_metrics["avg_flesch_grade_level"].toFixed(2)
          : "Not enough data to provide"}
      </p>
      <p className="text-center">
        <b>Average Grade Level of Text (Dale Chall Score): </b>
        {analysis.readability_metrics["avg_dale_chall_grade_level"] != -1
          ? analysis.readability_metrics["avg_dale_chall_grade_level"].toFixed(
              2
            )
          : "Not enough data to provide"}
      </p>
    </div>
  );
};
