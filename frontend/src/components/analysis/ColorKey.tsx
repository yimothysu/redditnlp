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
          <div className="bg-[#ff9898] h-5 w-18"></div>
          <div className="bg-[#ffc245] h-5 w-18"></div>
          <div className="bg-[#ffffac] h-5 w-18"></div>
          <div className="bg-[#d3ea84] h-5 w-18"></div>
          <div className="bg-[#91CC91] h-5 w-18"></div>
        </div>
        <p className="text-sm text-center">
          left = positive consensus, right = negative consensus
        </p>
      </div>
    );
  }

  return (
    <div className="ml-33 flex flex-col align-items">
      <h1 className="text-[13px] text-gray-500 pb-1 pl-35">consensus scale (negative to positive)</h1>
      <div className="flex">
      <h1 className="pr-3 text-gray-500 font-semibold pb-1">-1 (min)</h1>
      <ColorCodeBox backgroundColor="#ff9898"/>
      <ColorCodeBox backgroundColor="#ffc245"/>
      <ColorCodeBox backgroundColor="#ffffac"/>
      <ColorCodeBox backgroundColor="#d3ea84"/>
      <ColorCodeBox backgroundColor="#91CC91"/>
      <h1 className="pl-3 text-gray-500 font-semibold pb-1">1 (max)</h1>
    </div>
    </div>

  );
};
