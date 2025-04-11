interface AnalysisMenuProps {
  timeFilter: string;
  currentMenuItem: string;
  setCurrentMenuItem: (item: string) => void;
}

export const AnalysisMenu: React.FC<AnalysisMenuProps> = ({
  timeFilter,
  currentMenuItem,
  setCurrentMenuItem,
}) => {
  return (
    <div className="relative ml-7 inline-block text-left mb-60">
      <button className="font-semibold bg-indigo-600 text-white px-4 py-2 rounded">
        Menu
      </button>
      <div className="absolute mt-2 w-45 bg-white border border-gray-200 rounded shadow-lg z-10">
        <ul className="py-1">
          {timeFilter === "all" && (
            <>
              <li
                className={`px-4 py-2 cursor-pointer ${
                  currentMenuItem === "Ranking"
                    ? "bg-indigo-100 text-black"
                    : ""
                }`}
                onClick={() => setCurrentMenuItem("Ranking")}
              >
                Ranking
              </li>
              <hr />
            </>
          )}
          <li
            className={`px-4 py-2 cursor-pointer ${
              currentMenuItem === "Named Entities"
                ? "bg-indigo-100 text-black"
                : ""
            }`}
            onClick={() => setCurrentMenuItem("Named Entities")}
          >
            Named Entities
          </li>
          <hr />
          <li
            className={`px-4 py-2 cursor-pointer ${
              currentMenuItem === "Bi-Grams" ? "bg-indigo-100 text-black" : ""
            }`}
            onClick={() => setCurrentMenuItem("Bi-Grams")}
          >
            Bi-Grams
          </li>
          <hr />
          <li
            className={`px-4 py-2 cursor-pointer ${
              currentMenuItem === "Word Embeddings"
                ? "bg-indigo-100 text-black"
                : ""
            }`}
            onClick={() => setCurrentMenuItem("Word Embeddings")}
          >
            Word Embeddings
          </li>
          <hr />
          <li
            className={`px-4 py-2 cursor-pointer ${
              currentMenuItem === "Word Cloud" ? "bg-indigo-100 text-black" : ""
            }`}
            onClick={() => setCurrentMenuItem("Word Cloud")}
          >
            Word Cloud
          </li>
          <hr />
          <li
            className={`px-4 py-2 cursor-pointer ${
              currentMenuItem === "Readability Metrics"
                ? "bg-indigo-100 text-black"
                : ""
            }`}
            onClick={() => setCurrentMenuItem("Readability Metrics")}
          >
            Readability Metrics
          </li>
        </ul>
      </div>
    </div>
  );
};
