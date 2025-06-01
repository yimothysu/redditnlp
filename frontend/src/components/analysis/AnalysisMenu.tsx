interface TabProps {
  label: string;
  isActive: boolean;
  onClick: () => void;
}

const Tab: React.FC<TabProps> = ({ label, isActive, onClick }) => {
  return (
    <button
      className={`px-3 py-2 font-medium rounded-t-lg transition-colors duration-200 hover:cursor-pointer text-sm sm:text-base sm:px-4 whitespace-nowrap ${
        isActive
          ? "bg-[#fa6f4d] text-white"
          : "hover:bg-gray-100 text-gray-600"
      }`}
      onClick={onClick}
    >
      {label}
    </button>
  );
};

interface AnalysisMenuProps {
  timeFilter: string;
  currentMenuItem: string;
  setCurrentMenuItem: any;
}

// This component provides the menu tabs for the different metrics individuals can view
export const AnalysisMenu: React.FC<AnalysisMenuProps> = ({
  timeFilter,
  currentMenuItem,
  setCurrentMenuItem,
}) => {

  // List of different NLP analyses
  const tabs = [
    ...(timeFilter === "all" ? [{ label: "Ranking" }] : []),
    // { label: "Trends" },
    { label: "Named Entities" },
    { label: "Bi-Grams" },
    { label: "Word Embeddings" },
    { label: "Word Cloud" },
    { label: "Readability Metrics" },
  ];

  // For each NLP analysis, display tab
  return (
    <div className="mb-6 overflow-x-auto">
      <nav className="flex flex-wrap gap-2 border-b border-gray-200 pb-2 min-w-max sm:flex-nowrap">
        {tabs.map((tab) => (
          <Tab
            key={tab.label}
            label={tab.label}
            isActive={currentMenuItem === tab.label}
            onClick={() => setCurrentMenuItem(tab.label)}
          />
        ))}
      </nav>
    </div>
  );
};
