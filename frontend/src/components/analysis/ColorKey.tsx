interface ColorCodeBoxProps {
  backgroundColor: string;
  label: string;
}

const ColorCodeBox: React.FC<ColorCodeBoxProps> = ({
  backgroundColor,
  label,
}) => {
  return (
    <div
      className="flex items-center justify-center w-auto font-medium px-3 py-2 text-sm"
      style={{
        backgroundColor,
        textAlign: "center",
        maxWidth: "160px",
      }}
    >
      {label}
    </div>
  );
};

// This component displays the color code key for sentiment 
export const ColorKey: React.FC = () => {
  if (window.innerWidth < 500) {
    return (
      <div className="mt-8 mb-8">
        <div className="flex border-1 border">
          <div className="bg-[#AFE1AF] h-5 w-18"></div>
          <div className="bg-[#e2f4a5] h-5 w-18"></div>
          <div className="bg-[#FFFFC5] h-5 w-18"></div>
          <div className="bg-[#FFD580] h-5 w-18"></div>
          <div className="bg-[#ffb9b9] h-5 w-18"></div>
        </div>
        <p className="text-sm text-center">
          left = positive consensus, right = negative consensus
        </p>
      </div>
    );
  }

  return (
    <div className="flex justify-center m-8 rounded-md overflow-hidden">
      <ColorCodeBox backgroundColor="#AFE1AF" label="Very Positive Consensus" />
      <ColorCodeBox backgroundColor="#e2f4a5" label="Positive Consensus" />
      <ColorCodeBox backgroundColor="#FFFFC5" label="Neutral Consensus" />
      <ColorCodeBox backgroundColor="#FFD580" label="Negative Consensus" />
      <ColorCodeBox backgroundColor="#ffb9b9" label="Very Negative Consensus" />
    </div>
  );
};
