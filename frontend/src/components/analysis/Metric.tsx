interface MetricProps {
    name: string;
    metric: number;
  }
  
  export const Metric: React.FC<MetricProps> = ({ name, metric }) => {
    return (
      <>
        <div
            className="flex flex-row w-1/2 border bg-white border-gray-200 rounded-lg shadow-md m-2">
            <h3 className="text-center w-1/2 p-3 text-gray-500 font-semibold bg-gray-100 rounded-tl-lg rounded-bl-lg flex align-center items-center">
              {name}
            </h3>
            <div className="flex w-1/2 font-semibold pt-1 pb-1 pr-5 text-gray-400 pl-5 align-center items-center">
              {metric}
            </div>
        </div>
      </>
    );
  };