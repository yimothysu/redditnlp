interface MetricProps {
    name: string;
    metric: string | number;
  }
  
  // This component provides the style for readability metrics. It displays the name of the metric on the left and the numerical data on the right
  export const Metric: React.FC<MetricProps> = ({ name, metric }) => {
    return (
      <>
        <div
            className="flex flex-row sm:w-1/2 w-[90%] border bg-white border-gray-200 rounded-lg shadow-md m-2">
            <h3 className="text-center text-sm sm:w-3/8 w-3/4 p-3 text-gray-500 font-semibold bg-gray-100 rounded-tl-lg rounded-bl-lg flex align-center items-center">
              {name}
            </h3>
            <div className="flex font-semibold pt-1 pb-1 pr-5 text-gray-400 pl-5 align-center items-center text-sm">
              {metric}
            </div>
        </div>
      </>
    );
  };