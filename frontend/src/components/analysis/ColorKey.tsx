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
      className="w-auto font-medium border-1"
      style={{
        fontSize: "75%",
        backgroundColor,
        textAlign: "center",
        paddingLeft: "10px",
        paddingRight: "10px",
        paddingBottom: "7px",
        paddingTop: "7px",
        maxWidth: "160px",
      }}
    >
      {label}
    </div>
  );
};

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
    <div className="flex justify-center m-8">
      <ColorCodeBox backgroundColor="#AFE1AF" label="Very Positive Consensus" />
      <ColorCodeBox backgroundColor="#e2f4a5" label="Positive Consensus" />
      <ColorCodeBox backgroundColor="#FFFFC5" label="Neutral Consensus" />
      <ColorCodeBox backgroundColor="#FFD580" label="Negative Consensus" />
      <ColorCodeBox backgroundColor="#ffb9b9" label="Very Negative Consensus" />
    </div>
  );
};
