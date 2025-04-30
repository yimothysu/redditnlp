interface ColorCodeBoxProps {
  backgroundColor: string;
}

const ColorCodeBox: React.FC<ColorCodeBoxProps> = ({
  backgroundColor
}) => {
  return (
    <div
      className="flex items-center w-[70px] justify-center font-medium py-3 text-sm"
      style={{
        backgroundColor,
        textAlign: "center",
      }}
    >
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
    <div className="ml-50 flex">
      <ColorCodeBox backgroundColor="#ffb9b9"/>
      <ColorCodeBox backgroundColor="#FFD580"/>
      <ColorCodeBox backgroundColor="#FFFFC5"/>
      <ColorCodeBox backgroundColor="#e2f4a5"/>
      <ColorCodeBox backgroundColor="#AFE1AF"/>
    </div>
  );
};
