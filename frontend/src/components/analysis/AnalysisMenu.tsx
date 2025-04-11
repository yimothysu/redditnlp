interface TabProps {
  label: string;
  isActive: boolean;
  onClick: () => void;
}

const Tab: React.FC<TabProps> = ({ label, isActive, onClick }) => {
  return (
    <button
      className={`px-4 py-2 font-medium rounded-t-lg transition-colors duration-200 hover:cursor-pointer ${
        isActive
          ? "bg-indigo-600 text-white"
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
  setCurrentMenuItem: (item: string) => void;
}

export const AnalysisMenu: React.FC<AnalysisMenuProps> = ({
  timeFilter,
  currentMenuItem,
  setCurrentMenuItem,
}) => {
  const tabs = [
    ...(timeFilter === "all" ? [{ label: "Ranking" }] : []),
    { label: "Named Entities" },
    { label: "Bi-Grams" },
    { label: "Word Embeddings" },
    { label: "Word Cloud" },
    { label: "Readability Metrics" },
  ];

  return (
    <div className="mb-6">
      <nav className="flex space-x-2 border-b border-gray-200">
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
